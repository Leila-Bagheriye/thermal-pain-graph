"""
Shared model code.

STGraph: spatio-temporal graph over T frames x (G x G) spatial cells.
  * nodes  : one per cell per frame (T * 49 nodes for a 7 x 7 map)
  * edges  : 4-neighbour SPATIAL edges inside a frame
             + TEMPORAL edges linking the same cell across +/- win frames
             + self-loops; rows normalised (mean aggregation)
  * 2 residual graph-convolution layers
  * FACTORISED attention, which is the explanation:
        a_s : (B, 49)  spatial attention  -> 7 x 7 face map, per trial
        a_t : (B, T)   temporal attention -> which moments of the 5.5 s clip
  * use_graph=False keeps everything except the graph layers (ablation).
"""
import torch, torch.nn as nn, torch.nn.functional as F


def build_adj(T, G=7, win=2):
    N = T * G * G
    A = torch.zeros(N, N)
    node = lambda t, r, c: t * G * G + r * G + c
    for t in range(T):
        for r in range(G):
            for c in range(G):
                i = node(t, r, c)
                A[i, i] = 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if 0 <= r + dr < G and 0 <= c + dc < G:
                        A[i, node(t, r + dr, c + dc)] = 1
                for dt in range(-win, win + 1):
                    if dt and 0 <= t + dt < T:
                        A[i, node(t + dt, r, c)] = 1
    return A / A.sum(1, keepdim=True)


class STGraph(nn.Module):
    def __init__(self, C=512, T=28, G=7, H=128, win=2, use_graph=True, drop=0.3):
        super().__init__()
        self.use_graph = use_graph
        self.inp = nn.Sequential(nn.Linear(C, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(drop))
        self.pos_t = nn.Parameter(torch.zeros(1, T, 1, H))
        self.pos_s = nn.Parameter(torch.zeros(1, 1, G * G, H))
        self.register_buffer("A", build_adj(T, G, win))
        self.g1, self.g2 = nn.Linear(H, H), nn.Linear(H, H)
        self.n1, self.n2 = nn.LayerNorm(H), nn.LayerNorm(H)
        self.att_s = nn.Sequential(nn.Linear(H, H // 2), nn.Tanh(), nn.Linear(H // 2, 1))
        self.att_t = nn.Sequential(nn.Linear(H, H // 2), nn.Tanh(), nn.Linear(H // 2, 1))
        self.drop = nn.Dropout(drop)
        self.cls = nn.Linear(H, 2)

    def forward(self, x):                      # x: (B, T, G*G, C)
        B, T, N, _ = x.shape
        h = self.inp(x) + self.pos_t + self.pos_s
        if self.use_graph:
            h = h.reshape(B, T * N, -1)
            h = self.n1(h + F.gelu(self.A @ self.g1(h)))
            h = self.n2(h + F.gelu(self.A @ self.g2(h)))
            h = h.reshape(B, T, N, -1)
        h = self.drop(h)
        a_s = torch.softmax(self.att_s(h.mean(1)).squeeze(-1), -1)   # (B, N)
        z_t = (h * a_s[:, None, :, None]).sum(2)                      # (B, T, H)
        a_t = torch.softmax(self.att_t(z_t).squeeze(-1), -1)         # (B, T)
        z = (z_t * a_t[..., None]).sum(1)
        return self.cls(z), a_s, a_t
