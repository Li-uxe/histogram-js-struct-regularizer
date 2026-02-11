import datetime
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_settings import CFG

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

def run_degree_self_check():
    print("\n[Degree self-check]")

    # Toy undirected graph with known degrees:
    # edges: (0,1), (0,2), (1,2), (2,3)
    # expected degree: [2, 2, 3, 1]
    toy_edges = np.array([[0, 1], [0, 2], [1, 2], [2, 3]], dtype=np.int64)
    n = 4
    expected_deg = torch.tensor([2.0, 2.0, 3.0, 1.0], dtype=torch.float32)

    # Same undirected degree path used in train_once for deg_real.
    deg_real_like = torch.zeros(n, dtype=torch.float32)
    u = torch.tensor(toy_edges[:, 0], dtype=torch.long)
    v = torch.tensor(toy_edges[:, 1], dtype=torch.long)
    ones = torch.ones(len(toy_edges), dtype=torch.float32)
    deg_real_like.index_add_(0, u, ones)
    deg_real_like.index_add_(0, v, ones)

    # Undirected checks.
    no_self_loops = bool(np.all(toy_edges[:, 0] != toy_edges[:, 1]))
    undirected_unique = len({(min(int(a), int(b)), max(int(a), int(b))) for a, b in toy_edges}) == len(toy_edges)
    A = torch.zeros((n, n), dtype=torch.float32)
    for a, b in toy_edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    adj_symmetric = bool(torch.allclose(A, A.t()))
    deg_match_expected = bool(torch.allclose(deg_real_like, expected_deg))
    no_edge_double_count = abs(float(deg_real_like.sum().item()) - 2.0 * len(toy_edges)) < 1e-6

    # Predicted-degree path check: ensure diagonal is excluded.
    p_toy = torch.tensor([
        [0.9, 0.2, 0.1, 0.0],
        [0.2, 0.8, 0.3, 0.0],
        [0.1, 0.3, 0.7, 0.4],
        [0.0, 0.0, 0.4, 0.6],
    ], dtype=torch.float32)
    deg_with_diag = p_toy.sum(dim=1)
    deg_without_diag = deg_with_diag - p_toy.diagonal()
    expected_deg_without_diag = torch.tensor([0.3, 0.5, 0.8, 0.4], dtype=torch.float32)
    diag_excluded_ok = bool(torch.allclose(deg_without_diag, expected_deg_without_diag, atol=1e-6))
    prob_sum_rule_ok = bool(
        torch.isclose(
            deg_without_diag.sum(),
            2.0 * torch.triu(p_toy, diagonal=1).sum(),
            atol=1e-6,
            rtol=1e-6,
        ).item()
    )

    graph_mode = "undirected" if (undirected_unique and adj_symmetric) else "directed_or_inconsistent"
    print(f"graph_mode: {graph_mode}")
    print(f"toy_edges_no_self_loops: {no_self_loops}")
    print(f"adjacency_symmetric: {adj_symmetric}")
    print(f"deg_real_like: {deg_real_like.tolist()} expected: {expected_deg.tolist()} match={deg_match_expected}")
    print(f"undirected_no_double_count(sum_deg==2|E|): {no_edge_double_count}")
    print(f"deg_gen_diag_excluded_match: {diag_excluded_ok}")
    print(f"deg_gen_sum_rule(sum_deg==2*sum_upper): {prob_sum_rule_ok}")

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

def soft_histogram(x, bin_edges, sigma=0.55, tau=1.0, weights=None):
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    x = x.unsqueeze(1)
    c = centers.unsqueeze(0)
    sigma = max(float(sigma), 1e-6)
    # Gaussian KDE-style soft assignment gives non-zero mass to all bins.
    w = torch.exp(-0.5 * ((x - c) / sigma) ** 2)
    if weights is not None:
        weights = weights.clamp_min(0.0).unsqueeze(1)
        w = w * weights
    counts = w.sum(dim=0)
    p = counts / (counts.sum() + 1e-12)

    if tau is not None and tau > 0 and abs(tau - 1.0) > 1e-9:
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

def degree_mass_histogram(deg, bin_edges, sigma=0.55, tau=1.0, power=1.0):
    weights = deg.clamp_min(0.0)
    if abs(float(power) - 1.0) > 1e-9:
        weights = weights ** float(power)
    x = torch.log1p(deg)
    return soft_histogram(x, bin_edges, sigma=sigma, tau=tau, weights=weights)

def compute_degree_mass_js(deg_real, deg_gen, bin_edges, tau=1.0, hist_sigma=0.55, power=1.0):
    h_real = degree_mass_histogram(deg_real, bin_edges, sigma=hist_sigma, tau=tau, power=power)
    h_gen = degree_mass_histogram(deg_gen, bin_edges, sigma=hist_sigma, tau=tau, power=power)
    js = js_divergence(h_real, h_gen)
    return js, h_real, h_gen

def js_divergence(p, q, eps=1e-12):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    return 0.5 * kl_pm + 0.5 * kl_qm

def compute_L_struct_global(
    deg_real, deg_gen, bin_edges,
    tau=1.0, hist_sigma=0.55, use_soft=True,
    moment_weight=0.0, moment_beta=2.0
):
    x_real = torch.log1p(deg_real)
    x_gen  = torch.log1p(deg_gen)

    if use_soft:
        h_real = soft_histogram(x_real, bin_edges, sigma=hist_sigma, tau=tau)
        h_gen  = soft_histogram(x_gen,  bin_edges, sigma=hist_sigma, tau=tau)
    else:
        h_real = hard_histogram(x_real, bin_edges)
        h_gen  = hard_histogram(x_gen,  bin_edges)

    js = js_divergence(h_real, h_gen)
    _ = moment_beta  # kept for backward compatibility in CLI/config.
    m_mean = F.mse_loss(deg_gen.mean(), deg_real.mean())
    m_std = F.mse_loss(
        deg_gen.std(unbiased=False), deg_real.std(unbiased=False)
    )
    l_total = js + moment_weight * (m_mean + 0.5 * m_std)
    return l_total, js, m_mean, m_std, h_real.detach(), h_gen.detach()

def compute_hub_penalty(deg_real, deg_gen, top_ratio=0.01):
    k_real = max(1, int(len(deg_real) * top_ratio))
    k_gen = max(1, int(len(deg_gen) * top_ratio))
    real_top = torch.topk(deg_real, k_real).values
    gen_top = torch.topk(deg_gen, k_gen).values
    real_share = real_top.sum() / (deg_real.sum() + 1e-12)
    gen_share = gen_top.sum() / (deg_gen.sum() + 1e-12)
    # Bilateral penalty: over-concentration and over-diffusion are both penalized.
    return torch.abs(gen_share - real_share), real_share.detach(), gen_share.detach()

def calibrate_p_full_density(p_full, sample_logits, target_edge_rate, temp):
    pred_edge_rate = torch.sigmoid(sample_logits / max(1e-6, float(temp))).mean()
    target = torch.as_tensor(float(target_edge_rate), dtype=p_full.dtype, device=p_full.device)
    scale = (target / (pred_edge_rate.detach() + 1e-12)).clamp(0.1, 10.0)
    return (p_full * scale).clamp(0.0, 1.0)

def top_share(x, top_ratio=0.01):
    x = x.detach().cpu().numpy()
    n = len(x)
    k = max(1, int(n * top_ratio))
    idx = np.argsort(-x)
    return float(x[idx[:k]].sum() / (x.sum() + 1e-12))

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

def compute_struct_loss_scale(bce, l_struct):
    if not CFG.get("struct_loss_auto_scale", False):
        return 1.0
    raw = float(bce.detach().item()) / max(1e-8, float(l_struct.detach().item()))
    lo = float(CFG.get("struct_loss_scale_min", 0.25))
    hi = float(CFG.get("struct_loss_scale_max", 2.0))
    return float(np.clip(raw, lo, hi))

def grad_status(named_params, near_zero_thr=1e-8):
    none_names, small_names = [], []
    for name, p in named_params:
        if p.grad is None:
            none_names.append(name)
            continue
        gnorm = p.grad.detach().norm().item()
        if gnorm < near_zero_thr:
            small_names.append((name, gnorm))
    return none_names, small_names

def _float_for_filename(x):
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")

def build_run_csv_path(base_path, alpha_struct, seed):
    if base_path is None:
        return None
    base_path = str(base_path).strip()
    if base_path == "":
        return None
    root, ext = os.path.splitext(base_path)
    if ext == "":
        ext = ".csv"
    run_path = f"{root}_alpha{_float_for_filename(alpha_struct)}_seed{seed}{ext}"
    parent = os.path.dirname(run_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return run_path

def create_timestamp_run_dir(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, stamp)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(base_dir, f"{stamp}_{suffix:02d}")
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir

def _sanitize_filename_tag(tag):
    if tag is None:
        return ""
    s = str(tag).strip()
    if s == "":
        return ""
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)

def build_eval_artifact_prefix(run_dir, alpha_struct, seed, run_tag=None):
    if run_dir is None:
        return None
    run_dir = str(run_dir).strip()
    if run_dir == "":
        return None
    os.makedirs(run_dir, exist_ok=True)
    base = f"alpha{_float_for_filename(alpha_struct)}_seed{int(seed)}"
    safe_tag = _sanitize_filename_tag(run_tag)
    if safe_tag != "":
        base = f"{base}_{safe_tag}"
    return os.path.join(run_dir, base)

def save_eval_histograms(run_dir, alpha_struct, seed, result, run_tag=None):
    prefix = build_eval_artifact_prefix(run_dir, alpha_struct, seed, run_tag=run_tag)
    if prefix is None:
        return None, None

    npz_path = f"{prefix}_eval_histograms.npz"
    json_path = f"{prefix}_eval_histograms.json"

    np.savez(
        npz_path,
        bin_edges=np.asarray(result["bin_edges"], dtype=np.float64),
        soft_hist_real=np.asarray(result["h_real_soft"], dtype=np.float64),
        soft_hist_gen=np.asarray(result["h_gen_soft"], dtype=np.float64),
        soft_hist_real_mass=np.asarray(result["h_real_mass"], dtype=np.float64),
        soft_hist_gen_mass=np.asarray(result["h_gen_mass"], dtype=np.float64),
        hard_hist_real=np.asarray(result["h_real_hard"], dtype=np.float64),
        hard_hist_gen=np.asarray(result["h_gen_hard"], dtype=np.float64),
        js_soft=np.asarray([result["JS_soft"]], dtype=np.float64),
        js_hard=np.asarray([result["JS_hard"]], dtype=np.float64),
        js_mass=np.asarray([result["JS_mass"]], dtype=np.float64),
        hub_share_real=np.asarray([result["hub_top1%_share_real"]], dtype=np.float64),
        hub_share_gen=np.asarray([result["hub_top1%_share"]], dtype=np.float64),
    )

    payload = {
        "alpha_struct": float(alpha_struct),
        "seed": int(seed),
        "run_tag": _sanitize_filename_tag(run_tag),
        "bin_edges": np.asarray(result["bin_edges"], dtype=np.float64).tolist(),
        "soft_hist_real": np.asarray(result["h_real_soft"], dtype=np.float64).tolist(),
        "soft_hist_gen": np.asarray(result["h_gen_soft"], dtype=np.float64).tolist(),
        "soft_hist_real_mass": np.asarray(result["h_real_mass"], dtype=np.float64).tolist(),
        "soft_hist_gen_mass": np.asarray(result["h_gen_mass"], dtype=np.float64).tolist(),
        "hard_hist_real": np.asarray(result["h_real_hard"], dtype=np.float64).tolist(),
        "hard_hist_gen": np.asarray(result["h_gen_hard"], dtype=np.float64).tolist(),
        "js_soft": float(result["JS_soft"]),
        "js_hard": float(result["JS_hard"]),
        "js_mass": float(result["JS_mass"]),
        "hub_share_real_top1pct": float(result["hub_top1%_share_real"]),
        "hub_share_gen_top1pct": float(result["hub_top1%_share"]),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return npz_path, json_path

def save_eval_plots(run_dir, alpha_struct, seed, result, run_tag=None):
    prefix = build_eval_artifact_prefix(run_dir, alpha_struct, seed, run_tag=run_tag)
    if prefix is None:
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Plot] matplotlib unavailable, skipping plot export: {e}")
        return []

    bin_edges = np.asarray(result["bin_edges"], dtype=np.float64)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    bar_w = widths * 0.42
    if np.any(bar_w <= 0):
        bar_w = np.full_like(centers, 0.35, dtype=np.float64)

    real_soft = np.asarray(result["h_real_soft"], dtype=np.float64)
    gen_soft = np.asarray(result["h_gen_soft"], dtype=np.float64)
    real_hard = np.asarray(result["h_real_hard"], dtype=np.float64)
    gen_hard = np.asarray(result["h_gen_hard"], dtype=np.float64)

    soft_path = f"{prefix}_soft_hist.png"
    hard_path = f"{prefix}_hard_hist.png"
    hub_path = f"{prefix}_hub_share.png"

    def _plot_hist(real_hist, gen_hist, js_value, title, out_path):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(centers - 0.5 * bar_w, real_hist, width=bar_w, color="#1f77b4", alpha=0.8, label="real")
        ax.bar(centers + 0.5 * bar_w, gen_hist, width=bar_w, color="#ff7f0e", alpha=0.8, label="gen")
        ax.set_xlabel("log1p(degree)")
        ax.set_ylabel("Probability")
        ax.set_title(f"{title} | JS={js_value:.4f} | alpha={alpha_struct} seed={seed}")
        ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    _plot_hist(real_soft, gen_soft, float(result["JS_soft"]), "Soft Degree Histogram", soft_path)
    _plot_hist(real_hard, gen_hard, float(result["JS_hard"]), "Hard Degree Histogram", hard_path)

    hub_real = float(result["hub_top1%_share_real"])
    hub_gen = float(result["hub_top1%_share"])
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    bars = ax.bar(["real", "gen"], [hub_real, hub_gen], color=["#1f77b4", "#ff7f0e"], alpha=0.85)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top share")
    ax.set_title(f"Hub share top{int(CFG['hub_top_ratio'] * 100)}%")
    ax.grid(axis="y", alpha=0.25)
    for b in bars:
        h = float(b.get_height())
        y = min(0.98, h + 0.02)
        ax.text(b.get_x() + b.get_width() / 2.0, y, f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(hub_path, dpi=160)
    plt.close(fig)

    return [soft_path, hard_path, hub_path]
