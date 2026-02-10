import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Config (你只改这里)
# ============================================================
CFG = {
    # graph
    "n_nodes": 300,
    "n_comm": 3,
    "p_in": 0.03,
    "p_out": 0.001,
    "hub_ratio": 0.05,
    "hub_p": 0.40,
    "graph_seed": 1,

    # model / train
    "feat_dim": 64,
    "hid_dim": 128,
    "z_dim": 64,
    "epochs": 300,
    "lr": 1e-3,
    "neg_ratio": 5,
    "weight_decay": 5e-4,
    "grad_clip": 1.0,

    # struct regularizer schedule
    "warmup_frac": 0.25,
    "ramp_frac": 0.25,

    # histogram bins on log1p(deg)
    "bin_min": 0.0,
    "bin_max": 6.0,
    "n_bins": 12,

    # soft histogram temperature
    "tau": 0.25,

    # ✅ 结构项专用：避免 sigmoid 饱和
    "struct_logit_clip": 6.0,
    "struct_sigmoid_temp": 3.0,

    # ✅ debug/log
    "log_every": 25,
    "hub_top_ratio": 0.01,      # top1%
    "extra_hub_ratios": [0.05], # 额外打印 top5%（可选）

    # experiments
    "alphas": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "seeds": [1, 2, 3, 4, 5],
}

# ============================================================
# Utilities
# ============================================================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def auc_score_tie_aware(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)  # ascending
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    r = 1.0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (r + (r + (j - i)))
        ranks[i:j + 1] = avg_rank
        r += (j - i + 1)
        i = j + 1

    rank_sum_pos = ranks[y_sorted == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def average_precision_tie_safe(y_true, y_score, seed=0):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    rng = np.random.RandomState(seed)
    y_score = y_score + rng.uniform(-1e-12, 1e-12, size=y_score.shape)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    cum_pos = np.cumsum(y_sorted)
    precision = cum_pos / (np.arange(len(y_sorted)) + 1.0)
    ap = (precision * y_sorted).sum() / max(1, int(y_sorted.sum()))
    return float(ap)

def build_edge_set(edge_list):
    s = set()
    for u, v in edge_list:
        if u > v:
            u, v = v, u
        s.add((u, v))
    return s

def sample_negative_edges(n_nodes, pos_edge_set, num_samples, seed=0):
    rng = np.random.RandomState(seed)
    neg = []
    tries = 0
    max_tries = num_samples * 200
    while len(neg) < num_samples and tries < max_tries:
        u = int(rng.randint(0, n_nodes))
        v = int(rng.randint(0, n_nodes))
        if u == v:
            tries += 1
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in pos_edge_set:
            tries += 1
            continue
        neg.append((a, b))
        tries += 1
    return np.array(neg, dtype=np.int64)

# ============================================================
# Synthetic hub + communities graph
# ============================================================
def make_synthetic_hub_graph(
    n=300, n_comm=3, p_in=0.03, p_out=0.003,
    hub_ratio=0.02, hub_p=0.2, seed=1
):
    rng = np.random.RandomState(seed)
    comm = rng.randint(0, n_comm, size=n)
    n_hub = max(1, int(n * hub_ratio))
    hubs = rng.choice(n, size=n_hub, replace=False)

    edges = set()
    def add_edge(u, v):
        if u == v:
            return
        if u > v:
            u, v = v, u
        edges.add((u, v))

    for u in range(n):
        for v in range(u + 1, n):
            if comm[u] == comm[v]:
                if rng.rand() < p_in:
                    add_edge(u, v)
            else:
                if rng.rand() < p_out:
                    add_edge(u, v)

    for h in hubs:
        for v in range(n):
            if v != h and rng.rand() < hub_p:
                add_edge(h, v)

    edge_list = np.array(list(edges), dtype=np.int64)
    return edge_list, comm, hubs

# ============================================================
# Sparse normalized adjacency for GCN
# ============================================================
def make_sparse_adj(n_nodes, edge_list, add_self_loops=True, device="cpu"):
    rows, cols = [], []
    for u, v in edge_list:
        rows.extend([u, v])
        cols.extend([v, u])
    if add_self_loops:
        rows.extend(list(range(n_nodes)))
        cols.extend(list(range(n_nodes)))

    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(idx, vals, size=(n_nodes, n_nodes)).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg.clamp_min(1.0), -0.5)
    d_i = deg_inv_sqrt[idx[0]]
    d_j = deg_inv_sqrt[idx[1]]
    norm_vals = vals * d_i * d_j
    A_norm = torch.sparse_coo_tensor(idx, norm_vals, size=(n_nodes, n_nodes)).coalesce()
    return A_norm

# ============================================================
# GAE
# ============================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, A_norm, X):
        return torch.sparse.mm(A_norm, self.W(X))

class GAE(nn.Module):
    def __init__(self, in_dim, hid_dim=128, z_dim=64):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hid_dim)
        self.gcn2 = GCNLayer(hid_dim, z_dim)
        self.dec_bias = nn.Parameter(torch.zeros(1))

    def encode(self, A_norm, X):
        h = F.leaky_relu(self.gcn1(A_norm, X), negative_slope=0.1)
        z = self.gcn2(A_norm, h)
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def decode_logits_edges(self, z, edge_index):
        src, dst = edge_index[0], edge_index[1]
        logits = (z[src] * z[dst]).sum(dim=1) + self.dec_bias
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return logits.clamp(-20.0, 20.0)

    def decode_logits_full_raw(self, z):
        logits = z @ z.t() + self.dec_bias
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return logits

# ============================================================
# Histogram + JS
# ============================================================
def soft_histogram(x, bin_edges, tau=0.25):
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = (bin_edges[1] - bin_edges[0]).item()

    x = x.clamp(min=float(bin_edges[0].item()) + 1e-6,
                max=float(bin_edges[-1].item()) - 1e-6)

    x = x.unsqueeze(1)
    c = centers.unsqueeze(0)
    dist = torch.abs(x - c)
    w = F.relu(1.0 - dist / (width + 1e-12))
    counts = w.sum(dim=0)
    p = counts / (counts.sum() + 1e-12)

    if tau is not None and tau > 0:
        p = (p + 1e-12) ** (1.0 / max(1e-6, tau))
        p = p / p.sum()
    return p

def hard_histogram(x, bin_edges):
    x_np = x.detach().cpu().numpy()
    edges = bin_edges.detach().cpu().numpy()
    counts, _ = np.histogram(x_np, bins=edges)
    p = counts.astype(np.float64)
    p = p / max(1.0, p.sum())
    return torch.tensor(p, dtype=torch.float32, device=x.device)

def js_divergence(p, q, eps=1e-12):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    return 0.5 * kl_pm + 0.5 * kl_qm

def compute_L_struct_global(deg_real, deg_gen, bin_edges, tau=0.25, use_soft=True):
    x_real = torch.log1p(deg_real)
    x_gen  = torch.log1p(deg_gen)

    if use_soft:
        h_real = soft_histogram(x_real, bin_edges, tau=tau)
        h_gen  = soft_histogram(x_gen,  bin_edges, tau=tau)
    else:
        h_real = hard_histogram(x_real, bin_edges)
        h_gen  = hard_histogram(x_gen,  bin_edges)

    js = js_divergence(h_real, h_gen)
    return js, h_real.detach(), h_gen.detach()

def top_share(x, top_ratio=0.01):
    x = x.detach().cpu().numpy()
    n = len(x)
    k = max(1, int(n * top_ratio))
    idx = np.argsort(-x)
    return float(x[idx[:k]].sum() / (x.sum() + 1e-12))

# ============================================================
# Alpha schedule
# ============================================================
def alpha_schedule(ep, epochs, alpha_struct, warmup_frac=0.5, ramp_frac=0.25):
    warmup = int(epochs * warmup_frac)
    ramp = int(epochs * ramp_frac)
    if ep < warmup:
        return 0.0
    if ramp <= 0:
        return alpha_struct
    if ep < warmup + ramp:
        t = (ep - warmup) / max(1, ramp)
        return alpha_struct * float(t)
    return alpha_struct

# ============================================================
# Train / Eval
# ============================================================
def train_once(edge_list, n_nodes, feat_dim, hid_dim, z_dim,
               epochs, lr, neg_ratio, alpha_struct,
               warmup_frac, ramp_frac,
               weight_decay, grad_clip,
               seed, device):

    set_all_seeds(seed)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(edge_list))
    edge_list = edge_list[perm]
    n_train = int(0.8 * len(edge_list))
    train_edges = edge_list[:n_train]
    test_edges  = edge_list[n_train:]

    pos_all_set = build_edge_set(edge_list)
    A_norm = make_sparse_adj(n_nodes, train_edges, add_self_loops=True, device=device)

    model = GAE(in_dim=feat_dim, hid_dim=hid_dim, z_dim=z_dim).to(device)
    X_emb = nn.Embedding(n_nodes, feat_dim).to(device)
    nn.init.normal_(X_emb.weight, std=0.1)

    opt = torch.optim.Adam(
        list(model.parameters()) + list(X_emb.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # real degree on train graph
    deg_real = torch.zeros(n_nodes, device=device)
    u = torch.tensor(train_edges[:, 0], dtype=torch.long, device=device)
    v = torch.tensor(train_edges[:, 1], dtype=torch.long, device=device)
    ones_uv = torch.ones_like(u, dtype=torch.float32)
    deg_real.index_add_(0, u, ones_uv)
    deg_real.index_add_(0, v, ones_uv)

    hub_real_1 = top_share(deg_real, top_ratio=CFG["hub_top_ratio"])
    hub_real_extra = {r: top_share(deg_real, top_ratio=r) for r in CFG["extra_hub_ratios"]}

    # bins on log1p(deg)
    bin_edges = torch.linspace(CFG["bin_min"], CFG["bin_max"], steps=CFG["n_bins"] + 1, device=device)

    for ep in range(epochs):
        model.train()
        X = X_emb.weight
        z = model.encode(A_norm, X)

        # -------- BCE --------
        n_neg = neg_ratio * len(train_edges)
        neg_edges = sample_negative_edges(n_nodes, pos_all_set, n_neg, seed=seed + ep + 123)

        cand = np.vstack([train_edges, neg_edges])
        cand_t = torch.tensor(cand, dtype=torch.long, device=device)
        edge_index = torch.stack([cand_t[:, 0], cand_t[:, 1]], dim=0)

        y = torch.zeros(len(cand), device=device)
        y[:len(train_edges)] = 1.0

        logits = model.decode_logits_edges(z, edge_index)

        pos = float(y.sum().item())
        neg = float(len(y) - pos)
        pos_weight = torch.tensor([neg / max(1.0, pos)], device=device)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

        # -------- L_struct --------
        logits_full = model.decode_logits_full_raw(z)
        logits_full = logits_full.clamp(-CFG["struct_logit_clip"], CFG["struct_logit_clip"])
        p_full = torch.sigmoid(logits_full / CFG["struct_sigmoid_temp"])
        deg_gen = p_full.sum(dim=1) - p_full.diagonal()

        L_struct, _, _ = compute_L_struct_global(
            deg_real=deg_real,
            deg_gen=deg_gen,
            bin_edges=bin_edges,
            tau=CFG["tau"],
            use_soft=True
        )

        a_now = alpha_schedule(ep, epochs, alpha_struct, warmup_frac=warmup_frac, ramp_frac=ramp_frac)
        loss = bce + a_now * L_struct

        # ✅ sanity logs：看结构项有没有梯度、有多大
        if (ep % CFG["log_every"] == 0) or (ep == epochs - 1):
            g_bce = torch.autograd.grad(bce, z, retain_graph=True)[0]
            g_ls  = torch.autograd.grad(L_struct, z, retain_graph=True)[0]

            hub_gen_1 = top_share(deg_gen, top_ratio=CFG["hub_top_ratio"])
            msg = (f"[seed {seed} | alpha {alpha_struct} | ep {ep:04d}] "
                   f"a_now={a_now:.3g}  bce={bce.item():.4f}  Ls={L_struct.item():.4f}  loss={loss.item():.4f}  "
                   f"|g_bce|={g_bce.norm().item():.2e}  |g_Ls|={g_ls.norm().item():.2e}  "
                   f"deg_gen(mean/std/max)={deg_gen.mean().item():.2f}/{deg_gen.std().item():.2f}/{deg_gen.max().item():.2f}  "
                   f"hub1%(real/gen)={hub_real_1:.4f}/{hub_gen_1:.4f}")
            for r in CFG["extra_hub_ratios"]:
                msg += f"  hub{int(r*100)}%(real)={hub_real_extra[r]:.4f}"
            print(msg)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(X_emb.parameters()), grad_clip)
        opt.step()

    # ---------------- evaluation ----------------
    model.eval()
    with torch.no_grad():
        X = X_emb.weight
        z = model.encode(A_norm, X)

        # AUC/AP: test positives + sampled negatives
        pos_test = test_edges
        neg_test = sample_negative_edges(n_nodes, pos_all_set, num_samples=10 * len(pos_test), seed=seed + 999)

        cand = np.vstack([pos_test, neg_test])
        cand_t = torch.tensor(cand, dtype=torch.long, device=device)
        edge_index = torch.stack([cand_t[:, 0], cand_t[:, 1]], dim=0)

        y_np = np.zeros(len(cand), dtype=np.int64)
        y_np[:len(pos_test)] = 1

        logits = model.decode_logits_edges(z, edge_index)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)

        auc = auc_score_tie_aware(y_np, p)
        ap  = average_precision_tie_safe(y_np, p, seed=seed + 2026)

        # struct metrics (same pipeline)
        logits_full = model.decode_logits_full_raw(z)
        logits_full = logits_full.clamp(-CFG["struct_logit_clip"], CFG["struct_logit_clip"])
        p_full = torch.sigmoid(logits_full / CFG["struct_sigmoid_temp"])
        deg_gen = p_full.sum(dim=1) - p_full.diagonal()

        Ls, h_real, h_gen = compute_L_struct_global(
            deg_real=deg_real,
            deg_gen=deg_gen,
            bin_edges=bin_edges,
            tau=CFG["tau"],
            use_soft=False
        )

        hub_gen_1 = top_share(deg_gen, top_ratio=CFG["hub_top_ratio"])

    return {
        "AUC": float(auc),
        "AP": float(ap),
        "L_struct": float(Ls.item()),
        "hub_top1%_share": float(hub_gen_1),
        "hub_top1%_share_real": float(hub_real_1),
        "bin_edges": bin_edges.detach().cpu().numpy(),
        "h_real": h_real.detach().cpu().numpy(),
        "h_gen":  h_gen.detach().cpu().numpy(),
    }

# ============================================================
# Experiment runner
# ============================================================
def summarize(res_list, key):
    arr = np.array([r[key] for r in res_list], dtype=np.float64)
    return float(arr.mean()), float(arr.std())

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    edge_list, comm, hubs = make_synthetic_hub_graph(
        n=CFG["n_nodes"],
        n_comm=CFG["n_comm"],
        p_in=CFG["p_in"],
        p_out=CFG["p_out"],
        hub_ratio=CFG["hub_ratio"],
        hub_p=CFG["hub_p"],
        seed=CFG["graph_seed"],
    )

    print("Graph stats:",
          f"|V|={CFG['n_nodes']}",
          f"|E|={len(edge_list)}",
          f"hubs={len(hubs)}",
          f"hub_ratio={CFG['hub_ratio']}",
          f"hub_p={CFG['hub_p']}",
          f"p_in={CFG['p_in']}",
          f"p_out={CFG['p_out']}")

    all_alpha_results = {}

    for a in CFG["alphas"]:
        results = []
        for s in CFG["seeds"]:
            res = train_once(
                edge_list=edge_list,
                n_nodes=CFG["n_nodes"],
                feat_dim=CFG["feat_dim"],
                hid_dim=CFG["hid_dim"],
                z_dim=CFG["z_dim"],
                epochs=CFG["epochs"],
                lr=CFG["lr"],
                neg_ratio=CFG["neg_ratio"],
                alpha_struct=a,
                warmup_frac=CFG["warmup_frac"],
                ramp_frac=CFG["ramp_frac"],
                weight_decay=CFG["weight_decay"],
                grad_clip=CFG["grad_clip"],
                seed=s,
                device=device
            )
            results.append(res)

        all_alpha_results[a] = results

        auc_m, auc_s = summarize(results, "AUC")
        ap_m, ap_s   = summarize(results, "AP")
        ls_m, ls_s   = summarize(results, "L_struct")
        hub_m, hub_s = summarize(results, "hub_top1%_share")
        hubr_m, hubr_s = summarize(results, "hub_top1%_share_real")

        print(f"\n=== alpha={a} (mean±std over {len(CFG['seeds'])} seeds) ===")
        print(f"AUC: {auc_m:.4f} ± {auc_s:.4f}")
        print(f"AP : {ap_m:.4f} ± {ap_s:.4f}")
        print(f"L_struct(JS): {ls_m:.4f} ± {ls_s:.4f}")
        print(f"hub_top1%_share(gen): {hub_m:.4f} ± {hub_s:.4f}")
        print(f"hub_top1%_share(real): {hubr_m:.4f} ± {hubr_s:.4f}")

    # pick alpha: JS better, AUC not drop too much
    base_auc = summarize(all_alpha_results[0.0], "AUC")[0]
    best_alpha = 0.0
    best_js = 1e9
    for a in CFG["alphas"]:
        auc_m, _ = summarize(all_alpha_results[a], "AUC")
        js_m, _  = summarize(all_alpha_results[a], "L_struct")
        if auc_m >= base_auc - 0.01:
            if js_m < best_js:
                best_js = js_m
                best_alpha = a

    print(f"\nChosen alpha for PPT demo (AUC drop <=0.01): alpha={best_alpha}")

    r0 = all_alpha_results[0.0][0]
    rb = all_alpha_results[best_alpha][0]
    print("\n--- Histogram dump (one run) ---")
    print("bin_edges:", r0["bin_edges"])
    print("baseline h_real:", r0["h_real"])
    print("baseline h_gen :", r0["h_gen"])
    print("best    h_real:", rb["h_real"])
    print("best    h_gen :", rb["h_gen"])

if __name__ == "__main__":
    main()
