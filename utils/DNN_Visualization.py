"""
===============================================================================
VISUALIZATION INTEGRATION MODULE - Unified Plotting for Transfer Learning
===============================================================================

This module provides integrated visualization functions for transfer learning
results in nuclear decay prediction tasks.

Key Features:
- Six-panel RMS comparison plots with cross-validation statistics
- Seed-based scatter residual plots with flexible layout options
- Isotope scatter plots with UDL baseline comparison
- K-vs-RMS performance curves with uncertainty bands

Dependencies:
- matplotlib>=3.5.0
- pandas>=1.4.0
- numpy>=1.21.0

Author: [Your Name]
Date: 2026
"""

from utils.utils_DNN_Visualization import *
from utils.utils_DNN_Transfer import *
from utils.utils_Linear import *
import config
from pathlib import Path


def visualize_section(visualization_dir: Path = config.VISUALIZATION_DIR):
    """
    Integrated visualization function for transfer learning results.

    Generates:
    1. Six-panel comparison plots (RMS errors + residual distributions)
    2. Seed-based scatter residual plots (cluster/alpha modes)
    3. Isotope scatter plots with UDL baseline comparison
    4. K-vs-RMS performance curves (optional, uncomment for full run)

    Parameters
    ----------
    visualization_dir : Path, optional
        Base output directory for all visualizations (default: config.VISUALIZATION_DIR)
    """
    print(f'====================== STARTING TL VISUALIZATION ======================')
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # SECTION 1: SIX-PANEL COMPARISON PLOTS
    # =============================================================================
    print(f'\n====================== GENERATING SIX-PANEL COMPARISON ======================')
    config.SIX_PANEL_DIR.mkdir(parents=True, exist_ok=True)

    files_to_check = {
        'fold_rms_alpha': config.FOLD_RMS_ALPHA,
        'fold_rms_cluster': config.FOLD_RMS_CLUSTER,
        'fold_rms_combined': config.FOLD_RMS_COMBINED,
        'alpha_csv': config.ALPHA_CSV_PATH,
        'exploration_csv': config.EXPLORATION_PATH,
        'cluster_only_csv': config.CLUSTER_ONLY_EXTRAPOLATION_CSV,
        'combined_csv': config.COMBINED_EXTRAPOLATION_CSV
    }

    missing_files = []
    corrected_paths = {}

    for name, path in files_to_check.items():
        if not path.exists():
            print(f"[check] Warning: File not found at expected path: {path}")
            missing_files.append(name)
        else:
            corrected_paths[name] = str(path)
            print(f"[check] found {name} at: {path}")

    if missing_files:
        print(f"\n[check] Error: Cannot find the following required files: {', '.join(missing_files)}")

    plot_neural_network_evaluation_barchart(
        fold_rms_alpha=corrected_paths['fold_rms_alpha'],
        fold_rms_cluster=corrected_paths['fold_rms_cluster'],
        fold_rms_combined=corrected_paths['fold_rms_combined'],
        alpha_csv_path=corrected_paths['alpha_csv'],
        cluster_only_csv_path=corrected_paths['cluster_only_csv'],
        combined_csv_path=corrected_paths['combined_csv'],
        output_dir=str(config.SIX_PANEL_DIR)
    )
    print(f"[output] Six-panel residual comparison saved to: {config.SIX_PANEL_RESIDUAL_OUTPUT_PATH}")

    # =============================================================================
    # SECTION 2: SEED-BASED SCATTER RESIDUAL PLOTS
    # =============================================================================
    print(f'\n====================== GENERATING SEED SCATTER PLOT ======================')
    config.SEED_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    files_to_check = {
        'seed_results_cluster': config.SEED_RESULTS_CLUSTER_PATH,
        'seed_results_cluster_TL_ratio40': config.SEED_RATIO40_RESULTS_CLUSTER_TL_PATH,
        'seed_results_alpha': config.SEED_RESULTS_ALPHA_PATH,
    }

    missing_files = []
    corrected_paths = {}

    for name, path in files_to_check.items():
        if not path.exists():
            print(f"[check] Warning: File not found at expected path: {path}")
            missing_files.append(name)
        else:
            corrected_paths[name] = str(path)
            print(f"[check] found {name} at: {path}")

    if missing_files:
        print(f"\n[check] Error: Cannot find the following required files: {', '.join(missing_files)}")
        return

    # Plot cluster mode scatter (dual-file comparison, horizontal layout)
    plot_seed_scatter_residuals(
        seed_results_path=[
            corrected_paths['seed_results_cluster'],
            corrected_paths['seed_results_cluster_TL_ratio40']
        ],
        output_dir=str(config.SEED_COMPARISON_DIR),
        output_filename='seed_scatter_residuals_cluster.png',
        mode='cluster',
        layout='up'
    )

    print(f"[output] Seed scatter plot (cluster) saved to: {config.SEED_COMPARISON_DIR}")

    # =============================================================================
    # SECTION 3: SEED RMS COMPARISON
    # =============================================================================

    print(f'\n====================== GENERATING SEED RMS COMPARISON ======================')
    config.SEED_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    # Check file paths
    files_to_check = {
        'seed_results_cluster': config.SEED_RESULTS_CLUSTER_PATH,
        'seed_results_cluster_TL_ratio40': config.SEED_RATIO40_RESULTS_CLUSTER_TL_PATH
    }

    missing_files = []
    corrected_paths = {}

    for name, path in files_to_check.items():
        if not path.exists():
            print(f"[check] Warning: File not found at expected path: {path}")
            missing_files.append(name)
        else:
            corrected_paths[name] = str(path)
            print(f"[check] found {name} at: {path}")

    if missing_files:
        print(f"\n[check] Error: Cannot find the following required files: {', '.join(missing_files)}")
        return

    # Generate RMS comparison plot
    plot_seed_rms_comparison(
        csv_path=corrected_paths['seed_results_cluster'],
        output_dir=str(config.SEED_COMPARISON_DIR),
        output_filename='seed_rms_comparison.png',
        comparison_csv_path=corrected_paths['seed_results_cluster_TL_ratio40'],
        figsize=(26, 10),
        dpi=300
    )

    print(f"[output] Seed RMS comparison saved to: {config.SEED_COMPARISON_DIR}")

    # =============================================================================
    # SECTION 4: ISOTOPE SCATTER PLOTS WITH UDL BASELINE
    # =============================================================================
    print(f'\n====================== GENERATING ISOTOPE SCATTER PLOT ======================')
    config.ISOTOPE_SCATTER_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate UDL predictions
    try:
        udl_df = generate_udl_extrapolation(
            alpha_csv_path=str(config.ALPHA_DATA_PATH),
            cluster_csv_path=str(config.CLUSTER_DATA_PATH),
            exploration_csv_path=str(config.CLUSTER_DATA_EXPLORATION_PATH)
        )
        udl_df.to_csv(config.ISOTOPE_UDL_CSV, index=False)
        print(f"[output] UDL predictions saved to: {config.ISOTOPE_UDL_CSV}")
    except Exception as e:
        print(f"[check] UDL generation failed: {e}")
        raise

    # Step 2: Check plotting files
    plot_files = {
        'k10_csv': config.ISOTOPE_K10_CSV,
        'full_csv': config.ISOTOPE_FULL_CSV,
        'udl_csv': config.ISOTOPE_UDL_CSV
    }

    missing_plot_files = []
    corrected_plot_paths = {}

    for name, path in plot_files.items():
        if not path.exists():
            print(f"[check] Warning: File not found at expected path: {path}")
            missing_plot_files.append(name)
        else:
            corrected_plot_paths[name] = str(path)
            print(f"[check] found {name} at: {path}")

    if missing_plot_files:
        print(
            f"\n[check] Error: Cannot find the following required files for plotting: {', '.join(missing_plot_files)}")
        return

    # Step 3: Generate isotope scatter plot
    success = plot_isotope_scatter_combined(
        k10_csv_path=corrected_plot_paths.get('k10_csv', None),
        full_csv_path=corrected_plot_paths.get('full_csv', None),
        udl_csv_path=corrected_plot_paths.get('udl_csv', None),
        output_path=str(config.ISOTOPE_OUTPUT_PATH),
        plot_config=config.PLOT_CONFIG,
        plot_full=True  # Show both K=10 and full model predictions
    )

    if success:
        print(f"[output] Isotope scatter plot saved to: {config.ISOTOPE_OUTPUT_PATH}")
    else:
        print("[check] Isotope scatter plot generation failed.")

    # =============================================================================
    # SECTION 5: K-VS-RMS PERFORMANCE CURVES (OPTIONAL - UNCOMMENT FOR FULL RUN)
    # =============================================================================
    # Note: This section is commented out by default for faster testing.
    # Uncomment the entire block below when running the full visualization pipeline.

    # print(f'\n====================== EVALUATING SUBSETS ON UNIFIED MODELS ======================')
    # config.TL_400_SUBSETS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    # eval_config = config.TL_400_SUBSETS_EVAL_CONFIG
    # rms_summary_path = config.TRANSFER_RESULTS_DIR / "model_1_6" / "transfer_rms_summary.csv"
    #
    # # Validate RMS summary file existence
    # if not rms_summary_path.exists():
    #     print(f"[check] Error: RMS summary not found: {rms_summary_path}")
    # else:
    #     model_1_6_dir = str(config.TRANSFER_RESULTS_DIR / "model_1_6")
    #     if not Path(model_1_6_dir).exists():
    #         raise FileNotFoundError(f"[check] model_1_6 directory not found: {model_1_6_dir}\n")
    #
    #     try:
    #         results_df = evaluate_subsets_independent_training(
    #             rms_summary_path=str(rms_summary_path),
    #             alpha_data_path=str(config.ALPHA_DATA_PATH),
    #             model_1_6_dir=str(config.TRANSFER_RESULTS_DIR / "model_1_6"),
    #             output_csv_path=str(config.TL_400_SUBSETS_EVAL_OUTPUT_PATH),
    #             feature_cols=eval_config["feature_cols"],
    #             hidden_layers=eval_config["hidden_layers"],
    #             hidden_neurons=eval_config["hidden_neurons"],
    #             train_epochs=eval_config.get("train_epochs", 200),
    #             random_seed_base=eval_config["random_seed_base"]
    #         )
    #         print(f"[output] Results saved to: {config.TL_400_SUBSETS_EVAL_OUTPUT_PATH}")
    #
    #     except Exception as e:
    #         print(f"\n[check] CRITICAL ERROR during evaluation: {type(e).__name__}: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         results_df = None

    print(f'\n====================== GENERATING SHARED SUBSET RMS COMPARISON ======================')
    config.TL_SHARED_SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    vis_config = config.TL_SHARED_SUBSET_VIS_CONFIG
    base_model = vis_config["base_model"]
    compare_models = vis_config["compare_models"]
    best_n = vis_config["best_n"]

    extrapolation_csv = str(
        config.TL_400_SUBSETS_EVAL_OUTPUT_PATH) if config.TL_400_SUBSETS_EVAL_OUTPUT_PATH.exists() else None

    success = plot_k_vs_rms_shared_subsets(
        base_model=base_model,
        compare_models=compare_models,
        model_labels=vis_config["model_labels"],
        model_colors=vis_config["model_colors"],
        best_n=best_n,
        udl_csv_path=str(config.UDL_CSV_PATH),
        transfer_results_dir=str(config.TRANSFER_RESULTS_DIR),
        output_path=str(config.TL_SHARED_SUBSET_OUTPUT_PATH),
        title=vis_config["title"],
        show_uncertainty=vis_config["show_uncertainty"],
        figsize=(8, 6),
        dpi=300,
        extrapolation_rms_path=extrapolation_csv
    )

    if success:
        print(f"[output] RMS comparison saved to: {config.TL_SHARED_SUBSET_OUTPUT_PATH}")