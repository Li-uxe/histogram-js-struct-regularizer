import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_settings import CFG, STEP_METRIC_COLUMNS
from experiment_core import (
    GAE,
    _float_for_filename,
    _sanitize_filename_tag,
    alpha_schedule,
    auc_score_tie_aware,
    average_precision_tie_safe,
    build_edge_set,
    build_run_csv_path,
    calibrate_p_full_density,
    compute_L_struct_global,
    compute_degree_mass_js,
    compute_hub_penalty,
    grad_status,
    make_sparse_adj,
    make_synthetic_hub_graph,
    sample_negative_edges,
    save_eval_histograms,
    save_eval_plots,
    set_all_seeds,
    top_share,
    compute_struct_loss_scale,
)

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
        p_full = calibrate_p_full_density(
            p_full=p_full,
            sample_logits=logits_full,
            target_edge_rate=float(len(train_edges) / max(1.0, (n_nodes * (n_nodes - 1) / 2.0))),
            temp=CFG["struct_sigmoid_temp"],
        )
        density_penalty = float(CFG["density_penalty_weight"]) * F.mse_loss(
            p_full.mean(),
            torch.as_tensor(float(len(train_edges) / max(1.0, (n_nodes * (n_nodes - 1) / 2.0))), device=p_full.device),
        )
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

        js_mass, _, _ = compute_degree_mass_js(
            deg_real=deg_real,
            deg_gen=deg_gen,
            bin_edges=bin_edges,
            tau=CFG["tau"],
            hist_sigma=CFG["hist_sigma"],
            power=CFG["hub_mass_power"],
        )

        L_struct_total = L_struct + (float(CFG["hub_mass_js_weight"]) * js_mass) + density_penalty

        opt.zero_grad()
        L_struct_total.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(X_emb.parameters()), grad_clip)
        opt.step()

        if ep in (0, steps // 2, steps - 1):
            hub_gen_1 = top_share(deg_gen, top_ratio=CFG["hub_top_ratio"])
            print(
                f"[Sanity ep {ep:03d}] L_struct={L_struct_total.item():.4f} js={js_only.item():.4f} js_mass={js_mass.item():.4f} density={density_penalty.item():.4f} "
                f"deg_gen(mean/std/max)={deg_gen.mean().item():.2f}/{deg_gen.std().item():.2f}/{deg_gen.max().item():.2f} "
                f"hub1%(real/gen)={hub_real_1:.4f}/{hub_gen_1:.4f}"
            )

def train_once(edge_list, n_nodes, feat_dim, hid_dim, z_dim,
               epochs, lr, neg_ratio, alpha_struct,
               warmup_frac, ramp_frac,
               weight_decay, grad_clip,
               seed, device, metrics_csv_base="metrics/per_step_metrics.csv",
               run_dir=None, save_plots=False, run_tag=None):

    set_all_seeds(seed)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(edge_list))
    edge_list = edge_list[perm]
    n_train = int(0.8 * len(edge_list))
    train_edges = edge_list[:n_train]
    test_edges  = edge_list[n_train:]

    max_undirected_edges = n_nodes * (n_nodes - 1) / 2.0
    target_edge_rate = float(len(train_edges) / max(1.0, max_undirected_edges))

    pos_all_set = build_edge_set(edge_list)
    pos_train_set = build_edge_set(train_edges)
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
            neg_edges = sample_negative_edges(n_nodes, pos_train_set, n_neg, seed=seed + ep + 123)

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
            p_full = calibrate_p_full_density(
                p_full=p_full,
                sample_logits=logits_full,
                target_edge_rate=target_edge_rate,
                temp=CFG["struct_sigmoid_temp"],
            )
            density_penalty = float(CFG["density_penalty_weight"]) * F.mse_loss(
                p_full.mean(),
                torch.as_tensor(target_edge_rate, dtype=p_full.dtype, device=p_full.device),
            )
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

            js_mass, _, _ = compute_degree_mass_js(
                deg_real=deg_real,
                deg_gen=deg_gen,
                bin_edges=bin_edges,
                tau=CFG["tau"],
                hist_sigma=CFG["hist_sigma"],
                power=CFG["hub_mass_power"],
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
            hub_penalty = float(CFG["hub_penalty_weight"]) * hub_penalty_raw
            L_struct_total = L_struct + (float(CFG["hub_mass_js_weight"]) * js_mass) + hub_penalty + density_penalty
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
                    "density_penalty": float(density_penalty.item()),
                    "JS_soft": float(js_soft.item()),
                    "JS_mass": float(js_mass.item()),
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

            do_log = bool(CFG.get("verbose_train_logs", False)) and (
                (ep % CFG["log_every"] == 0) or (ep == epochs - 1)
            )
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
                       f"Ls(total/scaled/js/jsm)={L_struct_total.item():.4f}/{L_struct_scaled.item():.4f}/{js_soft.item():.4f}/{js_mass.item():.4f}  "
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
            p_full = calibrate_p_full_density(
                p_full=p_full,
                sample_logits=logits_full,
                target_edge_rate=target_edge_rate,
                temp=CFG["struct_sigmoid_temp"],
            )
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

            js_mass, h_real_mass, h_gen_mass = compute_degree_mass_js(
                deg_real=deg_real,
                deg_gen=deg_gen,
                bin_edges=bin_edges,
                tau=CFG["tau"],
                hist_sigma=CFG["hist_sigma"],
                power=CFG["hub_mass_power"],
            )

            hub_gen_1 = top_share(deg_gen, top_ratio=CFG["hub_top_ratio"])

        result = {
            "AUC": float(auc),
            "AP": float(ap),
            "L_struct": float(js_soft.item()),
            "L_struct_hard": float(js_hard.item()),
            "JS_soft": float(js_soft.item()),
            "JS_hard": float(js_hard.item()),
            "JS_mass": float(js_mass.item()),
            "hub_top1%_share": float(hub_gen_1),
            "hub_top1%_share_real": float(hub_real_1),
            "bin_edges": bin_edges.detach().cpu().numpy(),
            "h_real": h_real_soft.detach().cpu().numpy(),
            "h_gen": h_gen_soft.detach().cpu().numpy(),
            "h_real_soft": h_real_soft.detach().cpu().numpy(),
            "h_gen_soft": h_gen_soft.detach().cpu().numpy(),
            "h_real_hard": h_real_hard.detach().cpu().numpy(),
            "h_gen_hard": h_gen_hard.detach().cpu().numpy(),
            "h_real_mass": h_real_mass.detach().cpu().numpy(),
            "h_gen_mass": h_gen_mass.detach().cpu().numpy(),
        }

        if CFG.get("verbose_eval_logs", False):
            print(
                f"[Eval] seed={seed} alpha={alpha_struct} "
                f"AUC={result['AUC']:.4f} AP={result['AP']:.4f} "
                f"JS_soft={result['JS_soft']:.6f} JS_hard={result['JS_hard']:.6f} "
                f"JS_mass={result['JS_mass']:.6f}"
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
        if CFG.get("verbose_eval_logs", False) and npz_path is not None and json_path is not None:
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
            if CFG.get("verbose_eval_logs", False) and len(plot_paths) > 0:
                print(f"[Eval] saved plots: {', '.join(plot_paths)}")
        else:
            result["plot_paths"] = []

        return result
    finally:
        if metrics_file is not None:
            metrics_file.close()

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
        jsm_m, jsm_s = summarize(results, "JS_mass")
        hub_m, hub_s = summarize(results, "hub_top1%_share")
        hubr_m, hubr_s = summarize(results, "hub_top1%_share_real")
        hub_gap = np.array(
            [abs(r["hub_top1%_share_real"] - r["hub_top1%_share"]) for r in results],
            dtype=np.float64
        )
        hubg_m, hubg_s = float(hub_gap.mean()), float(hub_gap.std())
        rows.append({
            "alpha": float(a),
            "auc_mean": auc_m, "auc_std": auc_s,
            "ap_mean": ap_m, "ap_std": ap_s,
            "js_soft_mean": js_m, "js_soft_std": js_s,
            "js_mass_mean": jsm_m, "js_mass_std": jsm_s,
            "hub1pct_mean": hub_m, "hub1pct_std": hub_s,
            "hub1pct_real_mean": hubr_m, "hub1pct_real_std": hubr_s,
            "hub_gap_mean": hubg_m, "hub_gap_std": hubg_s,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| alpha | AUC(mean+/-std) | AP(mean+/-std) | JS-soft(mean+/-std) | JS-mass(mean+/-std) | hub1%_gen(mean+/-std) | hub1%_real(mean+/-std) | hub_gap(mean+/-std) |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['alpha']:.2f} | {r['auc_mean']:.4f}+/-{r['auc_std']:.4f} | "
                f"{r['ap_mean']:.4f}+/-{r['ap_std']:.4f} | "
                f"{r['js_soft_mean']:.4f}+/-{r['js_soft_std']:.4f} | "
                f"{r['js_mass_mean']:.4f}+/-{r['js_mass_std']:.4f} | "
                f"{r['hub1pct_mean']:.4f}+/-{r['hub1pct_std']:.4f} | "
                f"{r['hub1pct_real_mean']:.4f}+/-{r['hub1pct_real_std']:.4f} | "
                f"{r['hub_gap_mean']:.4f}+/-{r['hub_gap_std']:.4f} |\n"
            )

def main(metrics_csv_base="metrics/per_step_metrics.csv", run_dir=None, save_plots=False):
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

    if CFG.get("verbose_main_logs", False):
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
        jsm_m, jsm_s = summarize(results, "JS_mass")
        hub_m, hub_s = summarize(results, "hub_top1%_share")
        hubr_m, hubr_s = summarize(results, "hub_top1%_share_real")

        hub_gap = np.array(
            [abs(r["hub_top1%_share_real"] - r["hub_top1%_share"]) for r in results],
            dtype=np.float64
        )
        hubg_m, hubg_s = float(hub_gap.mean()), float(hub_gap.std())

        print(
            f"alpha={float(a):.2f} | "
            f"AUC={auc_m:.4f}+/-{auc_s:.4f} | "
            f"AP={ap_m:.4f}+/-{ap_s:.4f} | "
            f"JS_soft={ls_m:.4f}+/-{ls_s:.4f} | "
            f"JS_mass={jsm_m:.4f}+/-{jsm_s:.4f} | "
            f"hub1%_gen={hub_m:.4f}+/-{hub_s:.4f} | "
            f"hub1%_real={hubr_m:.4f}+/-{hubr_s:.4f} | "
            f"hub_gap={hubg_m:.4f}+/-{hubg_s:.4f}"
        )

    if run_dir is not None:
        table_path = os.path.join(run_dir, "alpha_summary_table.md")
        write_alpha_summary_table(all_alpha_results, table_path)
        if CFG.get("verbose_main_logs", False):
            print(f"[Summary] alpha table saved: {table_path}")
            print(f"\nArtifacts saved under: {run_dir}")
