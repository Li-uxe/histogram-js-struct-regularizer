import argparse
import csv
import datetime
import json
import os
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

    # soft histogram shaping
    "tau": 1.0,
    "hist_sigma": 0.55,
    "struct_moment_weight": 0.02,
    "struct_moment_beta": 2.0,
    "struct_loss_auto_scale": True,
    "struct_loss_scale_min": 0.25,
    "struct_loss_scale_max": 2.0,
    "hub_penalty_weight": 0.5,
    "hub_penalty_alpha_start": 5.0,

    # structure branch: smooth squash to avoid hard clamp zero-grad zones
    "struct_logit_clip": 6.0,
    "struct_sigmoid_temp": 2.0,

    # debug/log
    "log_every": 25,
    "hub_top_ratio": 0.01,      # top1%
    "extra_hub_ratios": [0.05], # extra print top5% (optional)
    "grad_near_zero_thr": 1e-8,
    "grad_report_max_names": 8,
    "run_degree_self_check": True,

    # quick sanity check: optimize only L_struct (BCE weight = 0)
    "run_struct_only_sanity": True,
    "struct_only_steps": 60,
    "struct_only_lr": 3e-3,

    # experiments
    "alphas": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "seeds": [1, 2, 3, 4, 5],
}

STEP_METRIC_COLUMNS = [
    "step",
    "bce",
    "L_struct_total",
    "L_struct_scaled",
    "hub_penalty",
    "JS_soft",
    "moment_mean",
    "moment_std",
    "deg_gen_mean",
    "deg_gen_std",
    "hub1pct_gen",
    "auc",
    "ap",
]

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Histogram + JS structure regularizer experiment")
    parser.add_argument(
        "--hist_sigma",
        type=float,
        default=CFG["hist_sigma"],
        help=f"KDE sigma for soft histogram (default: {CFG['hist_sigma']})",
    )
    parser.add_argument(
        "--struct_moment_weight",
        type=float,
        default=CFG["struct_moment_weight"],
        help=f"Weight for degree moment term (default: {CFG['struct_moment_weight']})",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=CFG["struct_sigmoid_temp"],
        help=f"Sigmoid temperature for structure branch (default: {CFG['struct_sigmoid_temp']})",
    )
    parser.add_argument(
        "--alpha_struct",
        type=float,
        default=None,
        help="If set, override CFG['alphas'] with a single alpha value",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="per_step_metrics.csv",
        help="Base path for per-step CSV metrics. Empty string disables CSV writing.",
    )
    parser.add_argument(
        "--run_sweep",
        action="store_true",
        help="Run built-in grid sweep and write summary CSV.",
    )
    parser.add_argument(
        "--sweep_epochs",
        type=int,
        default=200,
        help="Training epochs for each sweep configuration.",
    )
    parser.add_argument(
        "--sweep_seed",
        type=int,
        default=1,
        help="Fixed seed for all sweep configurations.",
    )
    parser.add_argument(
        "--sweep_results_csv",
        type=str,
        default="sweep_results.csv",
        help="Path to sweep summary CSV.",
    )
    parser.add_argument(
        "--sweep_min_auc",
        type=float,
        default=None,
        help="AUC threshold for selecting sweep results. Default: baseline_auc - 0.01.",
    )
    parser.add_argument(
        "--sweep_min_ap",
        type=float,
        default=None,
        help="AP threshold for selecting sweep results. Default: baseline_ap - 0.01.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save evaluation plots (soft/hard histograms and hub-share bar) to runs/<timestamp>/.",
    )
    return parser.parse_args(argv)


def apply_cli_overrides(args):
    CFG["hist_sigma"] = float(args.hist_sigma)
    CFG["struct_moment_weight"] = float(args.struct_moment_weight)
    CFG["struct_sigmoid_temp"] = float(args.temp)
    if args.alpha_struct is not None:
        CFG["alphas"] = [float(args.alpha_struct)]


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
def soft_histogram(x, bin_edges, sigma=0.55, tau=1.0):
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    x = x.unsqueeze(1)
    c = centers.unsqueeze(0)
    sigma = max(float(sigma), 1e-6)
    # Gaussian KDE-style soft assignment gives non-zero mass to all bins.
    w = torch.exp(-0.5 * ((x - c) / sigma) ** 2)
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
    m_mean = F.smooth_l1_loss(deg_gen.mean(), deg_real.mean(), beta=moment_beta)
    m_std = F.smooth_l1_loss(
        deg_gen.std(unbiased=False), deg_real.std(unbiased=False), beta=moment_beta
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
    # Only penalize over-concentrated hubs.
    return F.relu(gen_share - real_share), real_share.detach(), gen_share.detach()

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
        hard_hist_real=np.asarray(result["h_real_hard"], dtype=np.float64),
        hard_hist_gen=np.asarray(result["h_gen_hard"], dtype=np.float64),
        js_soft=np.asarray([result["JS_soft"]], dtype=np.float64),
        js_hard=np.asarray([result["JS_hard"]], dtype=np.float64),
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
        "hard_hist_real": np.asarray(result["h_real_hard"], dtype=np.float64).tolist(),
        "hard_hist_gen": np.asarray(result["h_gen_hard"], dtype=np.float64).tolist(),
        "js_soft": float(result["JS_soft"]),
        "js_hard": float(result["JS_hard"]),
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


def eval_auc_ap_from_z(model, z, edge_index, y_np, ap_seed):
    logits = model.decode_logits_edges(z, edge_index)
    p = torch.sigmoid(logits).detach().cpu().numpy()
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    auc = auc_score_tie_aware(y_np, p)
    ap = average_precision_tie_safe(y_np, p, seed=ap_seed)
    return float(auc), float(ap)


def run_struct_only_sanity(
    edge_list, n_nodes, feat_dim, hid_dim, z_dim,
    weight_decay, grad_clip, seed, device, steps=60, lr=3e-3
):
    set_all_seeds(seed)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(edge_list))
    edge_list = edge_list[perm]
    n_train = int(0.8 * len(edge_list))
    train_edges = edge_list[:n_train]

    A_norm = make_sparse_adj(n_nodes, train_edges, add_self_loops=True, device=device)

    model = GAE(in_dim=feat_dim, hid_dim=hid_dim, z_dim=z_dim).to(device)
    X_emb = nn.Embedding(n_nodes, feat_dim).to(device)
    nn.init.normal_(X_emb.weight, std=0.1)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(X_emb.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    deg_real = torch.zeros(n_nodes, device=device)
    u = torch.tensor(train_edges[:, 0], dtype=torch.long, device=device)
    v = torch.tensor(train_edges[:, 1], dtype=torch.long, device=device)
    ones_uv = torch.ones_like(u, dtype=torch.float32)
    deg_real.index_add_(0, u, ones_uv)
    deg_real.index_add_(0, v, ones_uv)

    bin_edges = torch.linspace(CFG["bin_min"], CFG["bin_max"], steps=CFG["n_bins"] + 1, device=device)
    hub_real_1 = top_share(deg_real, top_ratio=CFG["hub_top_ratio"])

    print("\n[Sanity] struct-only optimization (BCE weight = 0)")
    for ep in range(steps):
        model.train()
        z = model.encode(A_norm, X_emb.weight)

        logits_full = model.decode_logits_full_raw(z)
        s = float(CFG["struct_logit_clip"])
        logits_full = s * torch.tanh(logits_full / max(1e-6, s))
        p_full = torch.sigmoid(logits_full / CFG["struct_sigmoid_temp"])
        deg_gen = p_full.sum(dim=1) - p_full.diagonal()

        L_struct, js_only, _, _, _, _ = compute_L_struct_global(
            deg_real=deg_real,
            deg_gen=deg_gen,
            bin_edges=bin_edges,
            tau=CFG["tau"],
            hist_sigma=CFG["hist_sigma"],
            use_soft=True,
            moment_weight=CFG["struct_moment_weight"],
            moment_beta=CFG["struct_moment_beta"],
        )

        opt.zero_grad()
        L_struct.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(X_emb.parameters()), grad_clip)
        opt.step()

        if ep in (0, steps // 2, steps - 1):
            hub_gen_1 = top_share(deg_gen, top_ratio=CFG["hub_top_ratio"])
            print(
                f"[Sanity ep {ep:03d}] L_struct={L_struct.item():.4f} js={js_only.item():.4f} "
                f"deg_gen(mean/std/max)={deg_gen.mean().item():.2f}/{deg_gen.std().item():.2f}/{deg_gen.max().item():.2f} "
                f"hub1%(real/gen)={hub_real_1:.4f}/{hub_gen_1:.4f}"
            )

# ============================================================
# Train / Eval
# ============================================================
def train_once(edge_list, n_nodes, feat_dim, hid_dim, z_dim,
               epochs, lr, neg_ratio, alpha_struct,
               warmup_frac, ramp_frac,
               weight_decay, grad_clip,
               seed, device, metrics_csv_base="per_step_metrics.csv",
               run_dir=None, save_plots=False, run_tag=None):

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

    metrics_csv_path = build_run_csv_path(metrics_csv_base, alpha_struct, seed)
    print(
        f"\n[Run start] seed={seed} alpha_struct={alpha_struct} "
        f"hist_sigma={CFG['hist_sigma']} struct_moment_weight={CFG['struct_moment_weight']} "
        f"temp={CFG['struct_sigmoid_temp']} metrics_csv={metrics_csv_path or 'disabled'} "
        f"run_tag={_sanitize_filename_tag(run_tag) or 'none'}"
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

    # fixed eval set used both for per-step CSV and final report
    pos_test = test_edges
    neg_test = sample_negative_edges(n_nodes, pos_all_set, num_samples=10 * len(pos_test), seed=seed + 999)
    eval_cand = np.vstack([pos_test, neg_test])
    eval_cand_t = torch.tensor(eval_cand, dtype=torch.long, device=device)
    eval_edge_index = torch.stack([eval_cand_t[:, 0], eval_cand_t[:, 1]], dim=0)
    eval_y_np = np.zeros(len(eval_cand), dtype=np.int64)
    eval_y_np[:len(pos_test)] = 1

    metrics_file = None
    metrics_writer = None
    if metrics_csv_path is not None:
        metrics_file = open(metrics_csv_path, "w", newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=STEP_METRIC_COLUMNS)
        metrics_writer.writeheader()

    try:
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
            s = float(CFG["struct_logit_clip"])
            logits_full = s * torch.tanh(logits_full / max(1e-6, s))
            p_full = torch.sigmoid(logits_full / CFG["struct_sigmoid_temp"])
            deg_gen = p_full.sum(dim=1) - p_full.diagonal()

            L_struct, js_soft, moment_mean, moment_std, _, _ = compute_L_struct_global(
                deg_real=deg_real,
                deg_gen=deg_gen,
                bin_edges=bin_edges,
                tau=CFG["tau"],
                hist_sigma=CFG["hist_sigma"],
                use_soft=True,
                moment_weight=CFG["struct_moment_weight"],
                moment_beta=CFG["struct_moment_beta"],
            )

            with torch.no_grad():
                step_auc, step_ap = eval_auc_ap_from_z(
                    model=model,
                    z=z,
                    edge_index=eval_edge_index,
                    y_np=eval_y_np,
                    ap_seed=seed + 2026 + ep,
                )

            hub_penalty_raw, _, _ = compute_hub_penalty(
                deg_real=deg_real,
                deg_gen=deg_gen,
                top_ratio=CFG["hub_top_ratio"],
            )
            hub_alpha_gate = max(0.0, (float(alpha_struct) - float(CFG["hub_penalty_alpha_start"])) / max(1e-8, float(CFG["hub_penalty_alpha_start"])))
            hub_penalty = float(CFG["hub_penalty_weight"]) * hub_alpha_gate * hub_penalty_raw
            L_struct_total = L_struct + hub_penalty
            struct_scale = compute_struct_loss_scale(bce, L_struct_total)
            L_struct_scaled = struct_scale * L_struct_total

            deg_gen_mean = float(deg_gen.mean().item())
            deg_gen_std = float(deg_gen.std(unbiased=False).item())
            hub_gen_1 = top_share(deg_gen, top_ratio=CFG["hub_top_ratio"])
            if metrics_writer is not None:
                metrics_writer.writerow({
                    "step": ep,
                    "bce": float(bce.item()),
                    "L_struct_total": float(L_struct_total.item()),
                    "L_struct_scaled": float(L_struct_scaled.item()),
                    "hub_penalty": float(hub_penalty.item()),
                    "JS_soft": float(js_soft.item()),
                    "moment_mean": float(moment_mean.item()),
                    "moment_std": float(moment_std.item()),
                    "deg_gen_mean": deg_gen_mean,
                    "deg_gen_std": deg_gen_std,
                    "hub1pct_gen": float(hub_gen_1),
                    "auc": float(step_auc),
                    "ap": float(step_ap),
                })

            a_now = alpha_schedule(ep, epochs, alpha_struct, warmup_frac=warmup_frac, ramp_frac=ramp_frac)
            loss = bce + a_now * L_struct_scaled

            do_log = (ep % CFG["log_every"] == 0) or (ep == epochs - 1)
            if do_log:
                g_bce = torch.autograd.grad(bce, z, retain_graph=True)[0]
                g_ls  = torch.autograd.grad(L_struct_scaled, z, retain_graph=True)[0]

                g_bce_norm = g_bce.norm().item()
                g_ls_norm = g_ls.norm().item()
                g_ls_eff = a_now * g_ls_norm
                g_ratio = g_ls_eff / (g_bce_norm + 1e-12)

            opt.zero_grad()
            loss.backward()

            if do_log:
                named_params = list(model.named_parameters()) + [("X_emb.weight", X_emb.weight)]
                none_names, small_names = grad_status(
                    named_params, near_zero_thr=CFG["grad_near_zero_thr"]
                )
                max_names = int(CFG["grad_report_max_names"])
                small_head = ", ".join([f"{n}:{v:.1e}" for (n, v) in small_names[:max_names]])
                none_head = ", ".join(none_names[:max_names])

                msg = (f"[seed {seed} | alpha {alpha_struct} | ep {ep:04d}] "
                       f"a_now={a_now:.3g}  bce={bce.item():.4f}  "
                       f"Ls(base+hub/scaled/js)={L_struct_total.item():.4f}/{L_struct_scaled.item():.4f}/{js_soft.item():.4f}  "
                       f"mom(mean/std)={moment_mean.item():.4f}/{moment_std.item():.4f}  "
                       f"loss={loss.item():.4f}  "
                       f"|g_bce|={g_bce_norm:.2e}  |g_Ls|={g_ls_norm:.2e}  "
                       f"alpha*|g_Ls|={g_ls_eff:.2e}  (alpha*|g_Ls|)/|g_bce|={g_ratio:.2e}  "
                       f"deg_gen(mean/std/max)={deg_gen_mean:.2f}/{deg_gen_std:.2f}/{deg_gen.max().item():.2f}  "
                       f"hub1%(real/gen)={hub_real_1:.4f}/{hub_gen_1:.4f}  "
                       f"auc/ap={step_auc:.4f}/{step_ap:.4f}")
                for r in CFG["extra_hub_ratios"]:
                    msg += f"  hub{int(r*100)}%(real)={hub_real_extra[r]:.4f}"
                msg += f"  grad_none={len(none_names)}"
                if none_head:
                    msg += f" [{none_head}]"
                msg += f"  grad_small<{CFG['grad_near_zero_thr']:.0e}={len(small_names)}"
                if small_head:
                    msg += f" [{small_head}]"
                print(msg)

            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(X_emb.parameters()), grad_clip)
            opt.step()

        # ---------------- evaluation ----------------
        model.eval()
        with torch.no_grad():
            X = X_emb.weight
            z = model.encode(A_norm, X)

            auc, ap = eval_auc_ap_from_z(
                model=model, z=z, edge_index=eval_edge_index, y_np=eval_y_np, ap_seed=seed + 2026
            )

            # struct metrics (same pipeline)
            logits_full = model.decode_logits_full_raw(z)
            s = float(CFG["struct_logit_clip"])
            logits_full = s * torch.tanh(logits_full / max(1e-6, s))
            p_full = torch.sigmoid(logits_full / CFG["struct_sigmoid_temp"])
            deg_gen = p_full.sum(dim=1) - p_full.diagonal()

            _, js_soft, _, _, h_real_soft, h_gen_soft = compute_L_struct_global(
                deg_real=deg_real,
                deg_gen=deg_gen,
                bin_edges=bin_edges,
                tau=CFG["tau"],
                hist_sigma=CFG["hist_sigma"],
                use_soft=True,
                moment_weight=0.0
            )
            _, js_hard, _, _, h_real_hard, h_gen_hard = compute_L_struct_global(
                deg_real=deg_real,
                deg_gen=deg_gen,
                bin_edges=bin_edges,
                tau=CFG["tau"],
                hist_sigma=CFG["hist_sigma"],
                use_soft=False,
                moment_weight=0.0
            )

            hub_gen_1 = top_share(deg_gen, top_ratio=CFG["hub_top_ratio"])

        result = {
            "AUC": float(auc),
            "AP": float(ap),
            "L_struct": float(js_soft.item()),
            "L_struct_hard": float(js_hard.item()),
            "JS_soft": float(js_soft.item()),
            "JS_hard": float(js_hard.item()),
            "hub_top1%_share": float(hub_gen_1),
            "hub_top1%_share_real": float(hub_real_1),
            "bin_edges": bin_edges.detach().cpu().numpy(),
            "h_real": h_real_soft.detach().cpu().numpy(),
            "h_gen": h_gen_soft.detach().cpu().numpy(),
            "h_real_soft": h_real_soft.detach().cpu().numpy(),
            "h_gen_soft": h_gen_soft.detach().cpu().numpy(),
            "h_real_hard": h_real_hard.detach().cpu().numpy(),
            "h_gen_hard": h_gen_hard.detach().cpu().numpy(),
        }

        print(
            f"[Eval] seed={seed} alpha={alpha_struct} "
            f"AUC={result['AUC']:.4f} AP={result['AP']:.4f} "
            f"JS_soft={result['JS_soft']:.6f} JS_hard={result['JS_hard']:.6f}"
        )
        print(f"[Eval] soft_hist_real={result['h_real_soft']}")
        print(f"[Eval] soft_hist_gen ={result['h_gen_soft']}")
        print(f"[Eval] hard_hist_real={result['h_real_hard']}")
        print(f"[Eval] hard_hist_gen ={result['h_gen_hard']}")

        npz_path, json_path = save_eval_histograms(
            run_dir=run_dir,
            alpha_struct=alpha_struct,
            seed=seed,
            result=result,
            run_tag=run_tag,
        )
        result["eval_hist_npz"] = npz_path
        result["eval_hist_json"] = json_path
        if npz_path is not None and json_path is not None:
            print(f"[Eval] saved histograms: npz={npz_path} json={json_path}")

        if save_plots:
            plot_paths = save_eval_plots(
                run_dir=run_dir,
                alpha_struct=alpha_struct,
                seed=seed,
                result=result,
                run_tag=run_tag,
            )
            result["plot_paths"] = plot_paths
            if len(plot_paths) > 0:
                print(f"[Eval] saved plots: {', '.join(plot_paths)}")
        else:
            result["plot_paths"] = []

        return result
    finally:
        if metrics_file is not None:
            metrics_file.close()

# ============================================================
# Experiment runner
# ============================================================
def summarize(res_list, key):
    arr = np.array([r[key] for r in res_list], dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _print_sweep_table(rows):
    if len(rows) == 0:
        print("[Sweep] No configuration passed the AUC/AP thresholds.")
        return
    header = (
        f"{'hist_sigma':>10} {'moment_w':>10} {'alpha':>8} {'AUC':>8} {'AP':>8} "
        f"{'hub_gen':>10} {'hub_real':>10} {'hub_impr':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['hist_sigma']:>10.2f} {r['struct_moment_weight']:>10.2f} {r['alpha_struct']:>8.2f} "
            f"{r['auc']:>8.4f} {r['ap']:>8.4f} {r['hub1pct_gen']:>10.4f} "
            f"{r['hub1pct_real']:>10.4f} {r['hub1pct_improvement']:>10.4f}"
        )


def run_grid_sweep(args, run_dir=None, save_plots=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    edge_list, _, hubs = make_synthetic_hub_graph(
        n=CFG["n_nodes"],
        n_comm=CFG["n_comm"],
        p_in=CFG["p_in"],
        p_out=CFG["p_out"],
        hub_ratio=CFG["hub_ratio"],
        hub_p=CFG["hub_p"],
        seed=CFG["graph_seed"],
    )
    print(
        "Graph stats:",
        f"|V|={CFG['n_nodes']}",
        f"|E|={len(edge_list)}",
        f"hubs={len(hubs)}",
        f"hub_ratio={CFG['hub_ratio']}",
        f"hub_p={CFG['hub_p']}",
        f"p_in={CFG['p_in']}",
        f"p_out={CFG['p_out']}",
    )

    base_hist_sigma = CFG["hist_sigma"]
    base_moment_w = CFG["struct_moment_weight"]
    base_run_sanity = CFG.get("run_struct_only_sanity", False)
    CFG["run_struct_only_sanity"] = False

    try:
        # Baseline for "hub1pct_gen improvement": same setup with alpha_struct=0.0.
        CFG["hist_sigma"] = base_hist_sigma
        CFG["struct_moment_weight"] = base_moment_w
        baseline = train_once(
            edge_list=edge_list,
            n_nodes=CFG["n_nodes"],
            feat_dim=CFG["feat_dim"],
            hid_dim=CFG["hid_dim"],
            z_dim=CFG["z_dim"],
            epochs=int(args.sweep_epochs),
            lr=CFG["lr"],
            neg_ratio=CFG["neg_ratio"],
            alpha_struct=0.0,
            warmup_frac=CFG["warmup_frac"],
            ramp_frac=CFG["ramp_frac"],
            weight_decay=CFG["weight_decay"],
            grad_clip=CFG["grad_clip"],
            seed=int(args.sweep_seed),
            device=device,
            metrics_csv_base="",
            run_dir=run_dir,
            save_plots=save_plots,
            run_tag=f"sweep_baseline_hs{_float_for_filename(base_hist_sigma)}_mw{_float_for_filename(base_moment_w)}",
        )
        baseline_hub = float(baseline["hub_top1%_share"])

        min_auc = float(args.sweep_min_auc) if args.sweep_min_auc is not None else max(0.0, baseline["AUC"] - 0.01)
        min_ap = float(args.sweep_min_ap) if args.sweep_min_ap is not None else max(0.0, baseline["AP"] - 0.01)

        hist_sigmas = [0.5, 1.0, 2.0]
        moment_weights = [0.0, 0.1, 0.5, 1.0]
        alphas = [0.25, 0.5, 1.0]

        rows = []
        for hs in hist_sigmas:
            for mw in moment_weights:
                for a in alphas:
                    CFG["hist_sigma"] = float(hs)
                    CFG["struct_moment_weight"] = float(mw)
                    res = train_once(
                        edge_list=edge_list,
                        n_nodes=CFG["n_nodes"],
                        feat_dim=CFG["feat_dim"],
                        hid_dim=CFG["hid_dim"],
                        z_dim=CFG["z_dim"],
                        epochs=int(args.sweep_epochs),
                        lr=CFG["lr"],
                        neg_ratio=CFG["neg_ratio"],
                        alpha_struct=float(a),
                        warmup_frac=CFG["warmup_frac"],
                        ramp_frac=CFG["ramp_frac"],
                        weight_decay=CFG["weight_decay"],
                        grad_clip=CFG["grad_clip"],
                        seed=int(args.sweep_seed),
                        device=device,
                        metrics_csv_base="",
                        run_dir=run_dir,
                        save_plots=save_plots,
                        run_tag=f"sweep_hs{_float_for_filename(hs)}_mw{_float_for_filename(mw)}",
                    )
                    auc = float(res["AUC"])
                    ap = float(res["AP"])
                    hub_gen = float(res["hub_top1%_share"])
                    hub_real = float(res["hub_top1%_share_real"])
                    hub_impr = hub_gen - baseline_hub
                    pass_thr = (auc >= min_auc) and (ap >= min_ap)
                    rows.append({
                        "hist_sigma": float(hs),
                        "struct_moment_weight": float(mw),
                        "alpha_struct": float(a),
                        "epochs": int(args.sweep_epochs),
                        "seed": int(args.sweep_seed),
                        "auc": auc,
                        "ap": ap,
                        "hub1pct_gen": hub_gen,
                        "hub1pct_real": hub_real,
                        "hub1pct_improvement": float(hub_impr),
                        "hub_gap_abs": abs(hub_real - hub_gen),
                        "baseline_hub1pct_gen": baseline_hub,
                        "min_auc": float(min_auc),
                        "min_ap": float(min_ap),
                        "pass_threshold": int(pass_thr),
                    })

        results_csv = str(args.sweep_results_csv).strip()
        if results_csv == "":
            results_csv = "sweep_results.csv"
        csv_fields = [
            "hist_sigma", "struct_moment_weight", "alpha_struct", "epochs", "seed",
            "auc", "ap", "hub1pct_gen", "hub1pct_real",
            "hub1pct_improvement", "hub_gap_abs",
            "baseline_hub1pct_gen", "min_auc", "min_ap", "pass_threshold",
        ]
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        passing = [r for r in rows if r["pass_threshold"] == 1]
        passing_sorted = sorted(
            passing,
            key=lambda r: (r["hub1pct_improvement"], -r["hub_gap_abs"]),
            reverse=True
        )
        print(
            f"\n[Sweep] done. baseline_hub1pct_gen={baseline_hub:.4f} "
            f"min_auc={min_auc:.4f} min_ap={min_ap:.4f} "
            f"saved={results_csv}"
        )
        _print_sweep_table(passing_sorted)
    finally:
        CFG["hist_sigma"] = base_hist_sigma
        CFG["struct_moment_weight"] = base_moment_w
        CFG["run_struct_only_sanity"] = base_run_sanity


def write_alpha_summary_table(all_alpha_results, out_path):
    rows = []
    for a in sorted(all_alpha_results.keys(), key=float):
        results = all_alpha_results[a]
        auc_m, auc_s = summarize(results, "AUC")
        ap_m, ap_s = summarize(results, "AP")
        js_m, js_s = summarize(results, "L_struct")
        hub_m, hub_s = summarize(results, "hub_top1%_share")
        rows.append({
            "alpha": float(a),
            "auc_mean": auc_m, "auc_std": auc_s,
            "ap_mean": ap_m, "ap_std": ap_s,
            "js_soft_mean": js_m, "js_soft_std": js_s,
            "hub1pct_mean": hub_m, "hub1pct_std": hub_s,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| alpha | AUC(mean±std) | AP(mean±std) | JS-soft(mean±std) | hub1%(mean±std) |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['alpha']:.2f} | {r['auc_mean']:.4f}±{r['auc_std']:.4f} | "
                f"{r['ap_mean']:.4f}±{r['ap_std']:.4f} | "
                f"{r['js_soft_mean']:.4f}±{r['js_soft_std']:.4f} | "
                f"{r['hub1pct_mean']:.4f}±{r['hub1pct_std']:.4f} |\n"
            )

def main(metrics_csv_base="per_step_metrics.csv", run_dir=None, save_plots=False):
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

    if CFG.get("run_struct_only_sanity", False):
        run_struct_only_sanity(
            edge_list=edge_list,
            n_nodes=CFG["n_nodes"],
            feat_dim=CFG["feat_dim"],
            hid_dim=CFG["hid_dim"],
            z_dim=CFG["z_dim"],
            weight_decay=CFG["weight_decay"],
            grad_clip=CFG["grad_clip"],
            seed=CFG["seeds"][0],
            device=device,
            steps=int(CFG["struct_only_steps"]),
            lr=float(CFG["struct_only_lr"]),
        )

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
                device=device,
                metrics_csv_base=metrics_csv_base,
                run_dir=run_dir,
                save_plots=save_plots,
            )
            results.append(res)

        all_alpha_results[a] = results

        auc_m, auc_s = summarize(results, "AUC")
        ap_m, ap_s   = summarize(results, "AP")
        ls_m, ls_s   = summarize(results, "L_struct")
        lsh_m, lsh_s = summarize(results, "L_struct_hard")
        hub_m, hub_s = summarize(results, "hub_top1%_share")
        hubr_m, hubr_s = summarize(results, "hub_top1%_share_real")

        print(f"\n=== alpha={a} (mean±std over {len(CFG['seeds'])} seeds) ===")
        print(f"AUC: {auc_m:.4f} ± {auc_s:.4f}")
        print(f"AP : {ap_m:.4f} ± {ap_s:.4f}")
        print(f"L_struct(JS-soft): {ls_m:.4f} ± {ls_s:.4f}")
        print(f"L_struct(JS-hard): {lsh_m:.4f} ± {lsh_s:.4f}")
        print(f"hub_top1%_share(gen): {hub_m:.4f} ± {hub_s:.4f}")
        print(f"hub_top1%_share(real): {hubr_m:.4f} ± {hubr_s:.4f}")

    # pick alpha: JS better, AUC not drop too much
    base_alpha = 0.0 if 0.0 in all_alpha_results else CFG["alphas"][0]
    base_auc = summarize(all_alpha_results[base_alpha], "AUC")[0]
    best_alpha = base_alpha
    best_js = 1e9
    for a in CFG["alphas"]:
        auc_m, _ = summarize(all_alpha_results[a], "AUC")
        js_m, _  = summarize(all_alpha_results[a], "L_struct")
        if auc_m >= base_auc - 0.01:
            if js_m < best_js:
                best_js = js_m
                best_alpha = a

    print(f"\nChosen alpha for PPT demo (AUC drop <=0.01): alpha={best_alpha}")

    r0 = all_alpha_results[base_alpha][0]
    rb = all_alpha_results[best_alpha][0]
    print("\n--- Histogram dump (one run) ---")
    print("bin_edges:", r0["bin_edges"])
    print(f"baseline JS soft/hard: {r0['JS_soft']:.6f} / {r0['JS_hard']:.6f}")
    print("baseline soft h_real:", r0["h_real_soft"])
    print("baseline soft h_gen :", r0["h_gen_soft"])
    print("baseline hard h_real:", r0["h_real_hard"])
    print("baseline hard h_gen :", r0["h_gen_hard"])
    print(f"best JS soft/hard    : {rb['JS_soft']:.6f} / {rb['JS_hard']:.6f}")
    print("best    soft h_real:", rb["h_real_soft"])
    print("best    soft h_gen :", rb["h_gen_soft"])
    print("best    hard h_real:", rb["h_real_hard"])
    print("best    hard h_gen :", rb["h_gen_hard"])
    if run_dir is not None:
        table_path = os.path.join(run_dir, "alpha_summary_table.md")
        write_alpha_summary_table(all_alpha_results, table_path)
        print(f"[Summary] alpha table saved: {table_path}")
        print(f"\nArtifacts saved under: {run_dir}")

if __name__ == "__main__":
    cli_args = parse_args()
    apply_cli_overrides(cli_args)
    run_dir = create_timestamp_run_dir(base_dir="runs")
    print(f"[Artifacts] run_dir={run_dir}")
    if CFG.get("run_degree_self_check", False):
        run_degree_self_check()
    if cli_args.run_sweep:
        run_grid_sweep(cli_args, run_dir=run_dir, save_plots=bool(cli_args.save_plots))
    else:
        main(metrics_csv_base=cli_args.metrics_csv, run_dir=run_dir, save_plots=bool(cli_args.save_plots))
