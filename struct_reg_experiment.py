import argparse

from experiment_settings import CFG
from experiment_core import create_timestamp_run_dir, run_degree_self_check
from experiment_pipeline import main, run_grid_sweep

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
        default="metrics/per_step_metrics.csv",
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

if __name__ == "__main__":
    cli_args = parse_args()
    apply_cli_overrides(cli_args)
    run_dir = create_timestamp_run_dir(base_dir="runs")
    if CFG.get("verbose_main_logs", False):
        print(f"[Artifacts] run_dir={run_dir}")
    if CFG.get("run_degree_self_check", False):
        run_degree_self_check()
    if cli_args.run_sweep:
        run_grid_sweep(cli_args, run_dir=run_dir, save_plots=bool(cli_args.save_plots))
    else:
        main(metrics_csv_base=cli_args.metrics_csv, run_dir=run_dir, save_plots=bool(cli_args.save_plots))
