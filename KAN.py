# kan_alphazero.py
# Kolmogorov-Arnold Network (KAN) based AlphaZero-style backbone in PyTorch.
# Single-file model: KANLayer, stacked KAN blocks, policy & value heads.
# Practical approximation of KA representation: f(x) = sum_q g_q( sum_p phi_q(x_p) ),
# where phi_q is a small 1D network applied per scalar input and shared across positions.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerScalarInner(nn.Module):
    # Applies the same small 1D MLP to every scalar input x_p and emits Q outputs per scalar.
    def __init__(self, out_q, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_q)
        )

    def forward(self, x):  # x: (B, N) or (B, N, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # Flatten to (B*N,1) then map and reshape (B,N,Q)
        B, N, _ = x.shape
        y = self.net(x.view(B * N, 1))
        return y.view(B, N, -1)


class OuterPerQ(nn.Module):
    # Maps each scalar u_q to a vector of size out_dim (g_q)
    def __init__(self, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, u):  # u: (B, Q)
        B, Q = u.shape
        y = self.net(u.view(B * Q, 1))
        return y.view(B, Q, -1)  # (B,Q,out_dim)


class KANLayer(nn.Module):
    # One KAN layer implementing sum_q g_q(sum_p phi_q(x_p))
    def __init__(self, dim_in, dim_out, Q=64, inner_hidden=32, outer_hidden=64, layernorm=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.Q = Q
        # For efficiency we use the same PerScalarInner across all q and p (shared phi)
        self.inner = PerScalarInner(out_q=Q, hidden=inner_hidden)
        # For outer, produce dim_out per q, then sum over q -> final dim_out
        self.outer = OuterPerQ(out_dim=dim_out, hidden=outer_hidden)
        self.proj_in = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)
        self.layernorm = nn.LayerNorm(dim_out) if layernorm else None

    def forward(self, x):  # x: (B, dim_in)
        # project to working input dimension D (we will treat each of D coords as a scalar input)
        x_proj = self.proj_in(x)  # (B, D)
        B, D = x_proj.shape

        # Apply inner per-scalar MLP -> (B, D, Q)
        inner_out = self.inner(x_proj)  # (B, D, Q)

        # Sum over p (dimensions) -> u_q shape (B, Q)
        u = inner_out.sum(dim=1)  # (B, Q)

        # Apply outer per-q MLP => (B, Q, dim_out)
        outer_out = self.outer(u)  # (B, Q, dim_out)

        # Sum over q to get final vector (B, dim_out)
        summed = outer_out.sum(dim=1)  # (B, dim_out)

        if self.layernorm is not None:
            summed = self.layernorm(summed)
        return summed


class KANBlock(nn.Module):
    # A residual block of one or more KAN layers
    def __init__(self, dim, Q=64, inner_hidden=32, outer_hidden=64, depth=2, layernorm=True):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(KANLayer(dim, dim, Q=Q, inner_hidden=inner_hidden, outer_hidden=outer_hidden, layernorm=layernorm))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        for l in self.layers:
            y = l(out)
            out = self.act(out + y)  # residual
        return out


class AlphaZeroKANNet(nn.Module):
    # AlphaZero-like network using KAN blocks instead of convs.
    # input_channels: typical AZ uses 17, board_size default 8 -> input_vec_len = channels * 64
    def __init__(self,
                 input_channels=17,
                 board_size=8,
                 latent_dim=256,
                 num_blocks=6,
                 block_depth=2,
                 Q=64,
                 inner_hidden=32,
                 outer_hidden=64,
                 num_moves=4672):
        super().__init__()
        self.board_size = board_size
        self.input_channels = input_channels
        self.input_dim = input_channels * board_size * board_size
        self.latent_dim = latent_dim

        # Initial linear projection from flattened board to latent vector
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, latent_dim),
            nn.ReLU(inplace=True)
        )

        # Stacked KAN blocks
        self.blocks = nn.ModuleList([
            KANBlock(dim=latent_dim, Q=Q, inner_hidden=inner_hidden, outer_hidden=outer_hidden, depth=block_depth)
            for _ in range(num_blocks)
        ])

        # Policy head: project latent vector to move logits
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, num_moves)
        )

        # Value head: scalar between -1 and 1
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
            nn.Tanh()
        )

        self._init_weights()

    def forward(self, board):  # board: (B, C, H, W)
        B = board.shape[0]
        x = board.view(B, -1)
        x = self.input_proj(x)  # (B, latent_dim)
        for b in self.blocks:
            x = b(x)
        policy_logits = self.policy_head(x)  # (B, num_moves)
        value = self.value_head(x).squeeze(-1)  # (B,)
        return policy_logits, value

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # quick smoke test
    model = AlphaZeroKANNet(input_channels=17, board_size=8, latent_dim=256, num_blocks=4, block_depth=2, Q=64, num_moves=4672)
    dummy = torch.randn(2, 17, 8, 8)
    logits, value = model(dummy)
    print("policy logits:", logits.shape)  # expect (2, 4672)
    print("value:", value.shape)  # expect (2,)