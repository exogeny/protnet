import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from prot.ops import causal_conv1d_fn, causal_conv1d_update, mamba_chunk_scan_combined
from prot.ops._triton.layer_norm_gated import RMSNorm as RMSNormGated, LayerNorm
from prot.ops._triton.ssd_combined import mamba_split_conv1d_scan_combined
from prot.ops._triton.selective_state_update import selective_state_update
from prot.ops._triton.selective_scan import selective_scan_fn, mamba_inner_fn


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states, inference_params=None, seq_idx=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        if seq_idx is not None:
           raise RuntimeError('Mamba1 doesn\'t support the varlen sequences.')
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        x = causal_conv1d_update(
            x,
            conv_state,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.activation,
        )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        y = selective_state_update(
            ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
        )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Mamba2(nn.Module):
  def __init__(
      self,
      d_model,
      d_state=64,
      d_conv=4,
      conv_init=None,
      expand=2,
      headdim=128,
      ngroups=1,
      A_init_range=(1, 16),
      dt_min=0.001,
      dt_max=0.1,
      dt_init_floor=1e-4,
      dt_limit=(0.0, float("inf")),
      learnable_init_states=False,
      activation="swish",
      bias=False,
      conv_bias=True,
      # Fused kernel and sharding options
      chunk_size=256,
      use_mem_eff_path=True,
      layer_idx=None,  # Absorb kwarg for general module
  ):
    super().__init__()
    self.d_model = d_model
    self.d_state = d_state
    self.d_conv = d_conv
    self.conv_init = conv_init
    self.expand = expand
    self.d_inner = self.expand * self.d_model
    self.headdim = headdim
    self.ngroups = ngroups
    assert self.d_inner % self.headdim == 0
    self.nheads = self.d_inner // self.headdim
    self.dt_limit = dt_limit
    self.learnable_init_states = learnable_init_states
    self.activation = activation
    self.chunk_size = chunk_size
    self.use_mem_eff_path = use_mem_eff_path
    self.layer_idx = layer_idx

    # Order: [z, x, B, C, dt]
    d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
    self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

    conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
    self.conv1d = nn.Conv1d(
        in_channels=conv_dim,
        out_channels=conv_dim,
        bias=conv_bias,
        kernel_size=d_conv,
        groups=conv_dim,
        padding=d_conv - 1,
    )
    if self.conv_init is not None:
      nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

    if self.learnable_init_states:
      self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state))
      self.init_states._no_weight_decay = True

    self.act = nn.SiLU()

    # Initialize log dt bias
    dt = torch.exp(
        torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = torch.clamp(dt, min=dt_init_floor)
    # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    self.dt_bias = nn.Parameter(inv_dt)
    # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
    # name.endswith("bias") in param_grouping.py
    self.dt_bias._no_weight_decay = True

    # A parameter
    assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
    A = torch.zeros(self.nheads).uniform_(*A_init_range)
    A_log = torch.log(A)
    self.A_log = nn.Parameter(A_log)
    self.A_log._no_weight_decay = True

    # D "skip" parameter
    self.D = nn.Parameter(torch.ones(self.nheads))
    self.D._no_weight_decay = True

    # Extra normalization layer right before output projection
    assert RMSNormGated is not None
    self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)

    self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

  def forward(self, u, seq_idx=None):
    """
    u: (B, L, D)
    seq_idx: (batch, seqlen), int32
    Returns: same shape as u
    """
    batch, seqlen, dim = u.shape
    zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
    A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
    initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
    dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

    if self.use_mem_eff_path:
      # Fully fused path
      out = mamba_split_conv1d_scan_combined(
          zxbcdt,
          rearrange(self.conv1d.weight, "d 1 w -> d w"),
          self.conv1d.bias,
          self.dt_bias,
          A,
          D=self.D,
          chunk_size=self.chunk_size,
          seq_idx=seq_idx,
          activation=self.activation,
          rmsnorm_weight=self.norm.weight,
          rmsnorm_eps=self.norm.eps,
          outproj_weight=self.out_proj.weight,
          outproj_bias=self.out_proj.bias,
          headdim=self.headdim,
          ngroups=self.ngroups,
          norm_before_gate=False,
          initial_states=initial_states,
          **dt_limit_kwargs,
      )
    else:
      z, xBC, dt = torch.split(
          zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
      )
      dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
      assert self.activation in ["silu", "swish"]

      # 1D Convolution
      if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * ngroups * d_state)
        xBC = xBC[:, :seqlen, :]
      else:
        xBC = causal_conv1d_fn(
            x=xBC.contiguous().transpose(1, 2),
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
        ).transpose(1, 2)

      # Split into 3 main branches: X, B, C
      # These correspond to V, K, Q respectively in the SSM/attention duality
      x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
      y = mamba_chunk_scan_combined(
          rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
          dt,
          A,
          rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
          rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
          chunk_size=self.chunk_size,
          D=self.D,
          z=None,
          seq_idx=seq_idx,
          initial_states=initial_states,
          **dt_limit_kwargs,
      )
      y = rearrange(y, "b l h p -> b l (h p)")

      # Multiply "gate" branch and apply extra normalization layer
      y = self.norm(y, z)
      out = self.out_proj(y)
    return out
