# Shared experiment configuration and metric schema.

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
    "struct_moment_weight": 0.1,
    "struct_moment_beta": 2.0,
    "struct_loss_auto_scale": True,
    "struct_loss_scale_min": 0.25,
    "struct_loss_scale_max": 2.0,
    "hub_penalty_weight": 0.5,
    "hub_mass_js_weight": 0.5,
    "hub_mass_power": 1.0,
    "density_penalty_weight": 0.2,

    # structure branch: smooth squash to avoid hard clamp zero-grad zones
    "struct_logit_clip": 6.0,
    "struct_sigmoid_temp": 2.0,

    # debug/log
    "log_every": 25,
    "hub_top_ratio": 0.01,      # top1%
    "extra_hub_ratios": [0.05], # extra print top5% (optional)
    "grad_near_zero_thr": 1e-8,
    "grad_report_max_names": 8,
    "run_degree_self_check": False,
    "verbose_train_logs": False,
    "verbose_eval_logs": False,
    "verbose_main_logs": False,

    # quick sanity check: optimize only L_struct (BCE weight = 0)
    "run_struct_only_sanity": False,
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
    "density_penalty",
    "JS_soft",
    "JS_mass",
    "moment_mean",
    "moment_std",
    "deg_gen_mean",
    "deg_gen_std",
    "hub1pct_gen",
    "auc",
    "ap",
]
