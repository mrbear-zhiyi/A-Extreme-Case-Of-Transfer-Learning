from utils.utils_DNN_Transfer import *
from utils.utils_DNN_Transfer import _save_final_csv, _print_seed_summary
from utils.utils_Linear import *
import config
import time
import traceback
import numpy as np
import pandas as pd

"""
迁移函数集成调用函数
1、结合config调用迁移函数   ----   根据项目相对路径自适应输出结构
2、集成调用迁移函数         ----   简化形参配置及集成结果分析
"""


def transfer_models(
        output_root_dir: str | Path = config.TRANSFER_RESULTS_DIR,
        alpha_csv_path: str | Path = config.ALPHA_DATA_PATH,
        cluster_csv_path: str | Path = config.CLUSTER_DATA_PATH,
        full_cluster_csv: str | Path = config.CLUSTER_DATA_EXPLORATION_PATH
):
    # =============================================== 迁移学习初始配置模块 ==============================================
    output_root_dir = str(output_root_dir)
    alpha_csv_path = str(alpha_csv_path)
    cluster_csv_path = str(cluster_csv_path)
    full_cluster_csv = str(full_cluster_csv)
    os.makedirs(output_root_dir, exist_ok=True)
    start_time = time.time()

    # 从config获取配置（确保配置存在，不存在则使用默认值）
    full_training_model_name = getattr(config, 'FULL_TRAINING_MODEL', '1_6')
    enable_dual_extrapolation = getattr(config, 'ENABLE_DUAL_EXTRAPOLATION', True)

    print(f"Full training target model: {full_training_model_name}")
    if enable_dual_extrapolation:
        print("Dual extrapolation mode: ENABLED (full training + k=10 best subset)")
    else:
        print("Dual extrapolation mode: DISABLED (full training only)")

    print("UDL comparison...")
    udl_rms_alpha_cluster(
        alpha_csv_path=alpha_csv_path,
        cluster_csv_path=cluster_csv_path,
        output_dir=output_root_dir
    )

    print("\nCluster preprocessing...")
    Cluster = cluster_subsets(
        alpha_csv_path=alpha_csv_path,
        cluster_csv_path=cluster_csv_path,
    )



    cluster_data = pd.read_csv(cluster_csv_path)
    model_configs = []
    for model_name, freeze_first in config.TRANSFER_MODEL_CONFIGS.items():
        model_configs.append({
            "name": model_name,
            "path": str(config.get_model_path(model_name)),
            "freeze_first": freeze_first
        })
    total_models = len(model_configs)

    # =============================================== 迁移学习训练模块 ==============================================
    print(f"\n{'=' * 70}")
    print(f"Transfer learning for {total_models} alpha-trained models")
    print(f"{'=' * 70}")
    completed_models = 0
    all_results = {}
    successful_models = 0

    for config_item in model_configs:
        completed_models += 1
        model_name = config_item['name']
        model_start_time = time.time()

        print(f"\n[{completed_models}/{total_models}] Processing model: {model_name}")
        model_output_dir = os.path.join(output_root_dir, f"model_{model_name}")
        os.makedirs(model_output_dir, exist_ok=True)

        # 判断是否为全量训练目标模型
        full_training = (model_name == full_training_model_name)
        if full_training:
            print(f"  -> FULL TRAINING MODE: using all {len(cluster_data)} nuclei")
        else:
            print(f"  -> SUBSET TRAINING MODE: using preselected subsets (k=3~10)")

        try:
            best_exp_results, rms_df = transfer_learn_from_preselected_indices_all(
                alpha_model_path=config_item['path'],
                cluster_data=cluster_data,
                Cluster=Cluster,
                output_dir=model_output_dir,
                freeze_first=config_item['freeze_first'],
                patience=config.TRANSFER_TRAIN_CONFIG['patience'],
                lambda_init=config.TRANSFER_TRAIN_CONFIG['lambda_init'],
                overall_rms_summary_csv_path=os.path.join(model_output_dir, "k_avg_rms.csv"),
                full_training=full_training
            )

            all_results[model_name] = {
                'rms_summary': rms_df,
                'output_dir': model_output_dir,
                'processing_time': time.time() - model_start_time,
                'full_training': full_training
            }
            successful_models += 1
            mode_str = "FULL" if full_training else "SUBSET"
            print(f"[{completed_models}/{total_models}] Model {model_name} completed "
                  f"({mode_str}, {time.time() - model_start_time:.1f}s)")

        except Exception as e:
            print(f"[{completed_models}/{total_models}] Error in model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {
                'error': str(e),
                'output_dir': model_output_dir,
                'processing_time': time.time() - model_start_time,
                'full_training': full_training
            }

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"TRANSFER LEARNING COMPLETED")
    print(f"Total time: {total_time / 60:.1f} minutes | Success: {successful_models}/{total_models}")
    print(f"{'=' * 70}")

    # =============================================== 双重外推预测模块 ==============================================
    # 仅对配置的全量训练目标模型执行外推
    if full_training_model_name in all_results and 'error' not in all_results[full_training_model_name]:
        print(f"\n{'=' * 70}")
        print(f"EXTRAPOLATION PREDICTION FOR MODEL {full_training_model_name}")
        print(f"{'=' * 70}")

        model_base_dir = all_results[full_training_model_name]['output_dir']
        full_training_dir = os.path.join(model_base_dir, "full_training")

        # 检查全量训练模型是否存在
        if not os.path.exists(full_training_dir):
            print(f"Warning: Full training directory not found: {full_training_dir}")
        else:
            # ================================= 第一次外推：全量训练模型 =================================
            print("\n[1/2] Extrapolation using FULL TRAINING model...")
            try:
                # 创建独立输出目录
                full_extrap_dir = os.path.join(full_training_dir, "extrapolation_full")
                os.makedirs(full_extrap_dir, exist_ok=True)

                # 生成预测
                full_output_csv = os.path.join(full_extrap_dir, "cluster_prediction_summary.csv")
                full_summary_df = generate_cluster_prediction_summary(
                    model_dir=full_training_dir,
                    full_cluster_csv=full_cluster_csv,
                    output_csv=full_output_csv,
                    deduplicate=True
                )

                # 保存统计信息
                full_stats = {
                    'model_type': ['full_training'],
                    'total_nuclei': [len(full_summary_df)],
                    'certain_nuclei': [(full_summary_df['certainty'] == True).sum()],
                    'uncertain_nuclei': [(full_summary_df['certainty'] == False).sum()],
                    'mean_abs_diff_logT': [np.mean(np.abs(full_summary_df['difference_logT']))],
                    'rms_diff_logT': [np.sqrt(np.mean(full_summary_df['difference_logT'] ** 2))]
                }
                pd.DataFrame(full_stats).to_csv(
                    os.path.join(full_extrap_dir, "extrapolation_stats.csv"),
                    index=False
                )

                print(f"✓ Full training extrapolation completed")
                print(f"  Output: {full_output_csv}")
                print(
                    f"  Stats:  MAE={full_stats['mean_abs_diff_logT'][0]:.4f} | RMS={full_stats['rms_diff_logT'][0]:.4f}")

            except Exception as e:
                print(f"Error in full training extrapolation: {str(e)}")
                import traceback
                traceback.print_exc()

            # ================================= 第二次外推：k=10最佳子集模型（仅当启用双重外推时） ========================
            if enable_dual_extrapolation:
                print("\n[2/2] Extrapolation using BEST k=10 SUBSET model...")
                try:
                    # 从RMS汇总中找到k=10且overall_sigma_rms最小的模型
                    rms_summary_path = os.path.join(model_base_dir, 'transfer_rms_summary.csv')
                    if os.path.exists(rms_summary_path):
                        rms_df = pd.read_csv(rms_summary_path)
                        k10_mask = rms_df['k'] == 10

                        if k10_mask.any():
                            best_k10_row = rms_df[k10_mask].loc[rms_df[k10_mask]['overall_sigma_rms'].idxmin()]
                            k_val = int(best_k10_row['k'])
                            row_id = int(best_k10_row['row_id'])
                            best_k10_model_path = os.path.join(model_base_dir, f'transfer_k{k_val}_row{row_id}.pth')

                            if os.path.exists(best_k10_model_path):
                                # 创建独立输出目录
                                k10_extrap_dir = os.path.join(model_base_dir, "extrapolation_k10_best")
                                os.makedirs(k10_extrap_dir, exist_ok=True)

                                # 生成预测（使用与全量训练相同的函数，仅模型路径不同）
                                k10_output_csv = os.path.join(k10_extrap_dir, "cluster_prediction_summary.csv")
                                k10_summary_df = generate_cluster_prediction_summary(
                                    model_dir=model_base_dir,  # 注意：此处传入base_dir，函数内部会自动选择k=10最佳模型
                                    full_cluster_csv=full_cluster_csv,
                                    output_csv=k10_output_csv,
                                    deduplicate=True
                                )

                                # 保存统计信息
                                k10_stats = {
                                    'model_type': ['k10_best_subset'],
                                    'k_value': [k_val],
                                    'row_id': [row_id],
                                    'train_rms': [best_k10_row['train_sigma_rms']],
                                    'test_rms': [best_k10_row['test_sigma_rms']],
                                    'overall_rms': [best_k10_row['overall_sigma_rms']],
                                    'total_nuclei': [len(k10_summary_df)],
                                    'certain_nuclei': [(k10_summary_df['certainty'] == True).sum()],
                                    'uncertain_nuclei': [(k10_summary_df['certainty'] == False).sum()],
                                    'mean_abs_diff_logT': [np.mean(np.abs(k10_summary_df['difference_logT']))],
                                    'rms_diff_logT': [np.sqrt(np.mean(k10_summary_df['difference_logT'] ** 2))]
                                }
                                pd.DataFrame(k10_stats).to_csv(
                                    os.path.join(k10_extrap_dir, "extrapolation_stats.csv"),
                                    index=False
                                )

                                print(f"✓ k=10 best subset extrapolation completed")
                                print(
                                    f"  Model: k={k_val}, row={row_id} | Train RMS={best_k10_row['train_sigma_rms']:.4f}")
                                print(f"  Output: {k10_output_csv}")
                                print(
                                    f"  Stats:  MAE={k10_stats['mean_abs_diff_logT'][0]:.4f} | RMS={k10_stats['rms_diff_logT'][0]:.4f}")
                            else:
                                print(f"Warning: Best k=10 model file not found: {best_k10_model_path}")
                        else:
                            print("Warning: No k=10 training results found in RMS summary")
                    else:
                        print(f"Warning: RMS summary not found: {rms_summary_path}")
                except Exception as e:
                    print(f"Error in k=10 best subset extrapolation: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # ================================= 生成双重外推对比报告 =================================
            if enable_dual_extrapolation:
                try:
                    full_stats_path = os.path.join(full_extrap_dir, "extrapolation_stats.csv")
                    k10_stats_path = os.path.join(k10_extrap_dir, "extrapolation_stats.csv")

                    if os.path.exists(full_stats_path) and os.path.exists(k10_stats_path):
                        full_stats_df = pd.read_csv(full_stats_path)
                        k10_stats_df = pd.read_csv(k10_stats_path)

                        # 合并对比报告
                        comparison_df = pd.DataFrame({
                            'Model_Type': ['Full_Training', 'k10_Best_Subset'],
                            'MAE_diff_logT': [
                                full_stats_df['mean_abs_diff_logT'].iloc[0],
                                k10_stats_df['mean_abs_diff_logT'].iloc[0]
                            ],
                            'RMS_diff_logT': [
                                full_stats_df['rms_diff_logT'].iloc[0],
                                k10_stats_df['rms_diff_logT'].iloc[0]
                            ],
                            'Certain_Nuclei': [
                                full_stats_df['certain_nuclei'].iloc[0],
                                k10_stats_df['certain_nuclei'].iloc[0]
                            ],
                            'Uncertain_Nuclei': [
                                full_stats_df['uncertain_nuclei'].iloc[0],
                                k10_stats_df['uncertain_nuclei'].iloc[0]
                            ]
                        })

                        comparison_path = os.path.join(model_base_dir, "extrapolation_comparison.csv")
                        comparison_df.to_csv(comparison_path, index=False)

                        print(f"\n✓ Extrapolation comparison report saved to: {comparison_path}")
                        print("\nComparison Summary:")
                        print(comparison_df.to_string(index=False))
                except Exception as e:
                    print(f"Warning: Failed to generate comparison report: {str(e)}")
    else:
        print(f"\nSkipping extrapolation: model {full_training_model_name} not successfully trained")

    # =============================================== 迁移学习结果整合模块 ==============================================
    summary_data = []
    fixed_sizes = (3, 4, 5, 6, 7, 8, 9, 10)
    for model_name, result in all_results.items():
        if 'error' not in result:
            rms_df = result['rms_summary']
            k_avg_rms = {}

            if not result.get('full_training', False) and rms_df is not None and not rms_df.empty:
                for k in fixed_sizes:
                    k_mask = rms_df['k'] == k
                    if isinstance(k_mask, pd.Series) and k_mask.any():
                        best_k_df = rms_df[k_mask].nsmallest(3, 'overall_sigma_rms')
                        k_avg_rms[k] = best_k_df['overall_sigma_rms'].mean()

            if result.get('full_training', False) and rms_df is not None and not rms_df.empty:
                k_avg_rms['full'] = rms_df['overall_sigma_rms'].iloc[0]

            if k_avg_rms:
                summary_data.append({
                    'model_name': model_name,
                    'layers': int(model_name.split('_')[0]),
                    'neurons': int(model_name.split('_')[1]),
                    'avg_rms': np.mean(list(k_avg_rms.values())),
                    'training_mode': 'full' if result.get('full_training', False) else 'subset',
                    'is_full_training_target': model_name == full_training_model_name,
                    'processing_time': result['processing_time']
                })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_root_dir, "all_alpha_models_transfer_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        if not summary_df.empty:
            best_model = summary_df.loc[summary_df['avg_rms'].idxmin()]
            print(f"\nBest performing model: {best_model['model_name']} "
                  f"(avg_rms: {best_model['avg_rms']:.4f})")

            target_model = summary_df[summary_df['is_full_training_target']].iloc[0] if not summary_df[
                summary_df['is_full_training_target']].empty else None
            if target_model is not None:
                print(f"Full training target ({full_training_model_name}): avg_rms = {target_model['avg_rms']:.4f}")

    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Transfer learning results: {output_root_dir}")
    if full_training_model_name in all_results and 'error' not in all_results[full_training_model_name]:
        base_dir = all_results[full_training_model_name]['output_dir']
        print(f"Full training extrapolation: {os.path.join(base_dir, 'full_training', 'extrapolation_full')}")
        if enable_dual_extrapolation:
            print(f"k=10 best subset extrapolation: {os.path.join(base_dir, 'extrapolation_k10_best')}")
            print(f"Comparison report: {os.path.join(base_dir, 'extrapolation_comparison.csv')}")
    print(f"{'=' * 70}")

    return all_results

def Batch_transfer_learning(
        freeze_first: int = None,
        max_epochs: int = 200,
        patience: int = 100,
        lambda_init: float = 1e-3,
        top_k_models: int = 10,
        alpha_model_patterns: list = None,
        group_size: int = 27,
        skip_transfer: bool = True,
        overall_rms_threshold: float = 1.5,
):
    """
    批量迁移学习调用函数（支持完整/筛选双模式）

    模式说明：
    - skip_transfer=False: 执行迁移学习 → 四种筛选方案
    - skip_transfer=True: 跳过迁移学习，直接读取已有结果 → 四种筛选方案

    筛选条件：
    - overall_rms ≤ overall_rms_threshold (默认 1.5)

    四条独立筛选路径：
    - Path A: 基于模型参数距离筛选 → transfer_seeds_model.csv
    - Path B: 基于紧凑簇的 ratio 相似性筛选 → transfer_seeds_ratio.csv
    - Path C: 基于测试核 ratio 均值距 0 距离筛选 → transfer_seeds_ratio_mean.csv
    - Path D: 基于层次聚类筛选 → transfer_seeds_Kmean.csv
    """
    # ========== 路径配置 ==========
    output_base = Path(config.MULTI_SEED_OUTPUT_DIR)
    transfer_output_base = Path(config.TL_MULTI_SEED_OUTPUT_DIR)
    train_csv_path = Path(config.CLUSTER_TRAIN_DATA_PATH)
    test_csv_path = Path(config.CLUSTER_TEST_DATA_PATH)

    # 文件检查
    for p in [train_csv_path, test_csv_path, output_base]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    transfer_output_base.mkdir(parents=True, exist_ok=True)

    # ========== 步骤一：获取所有种子结果 ==========
    all_results = []
    skipped_by_rms = 0

    if skip_transfer:
        # ========== 筛选模式：读取已有结果 ==========
        print("\n[Mode] SKIP TRANSFER - Loading existing results...")

        seed_folders = sorted([d for d in transfer_output_base.iterdir()
                               if d.is_dir() and d.name.startswith('seed')])

        print(f"[Mode] Found {len(seed_folders)} seed folders")

        for seed_folder in seed_folders:
            try:
                random_seed = int(seed_folder.name.replace('seed', ''))
            except ValueError:
                continue

            # 读取 RMS 指标
            rms_path = seed_folder / 'rms_metrics.csv'
            if not rms_path.exists():
                print(f"[Warning] No rms_metrics.csv for seed {random_seed}, skipping...")
                skipped_by_rms += 1
                continue

            rms_dict = pd.read_csv(rms_path).iloc[0].to_dict()

            overall_rms = rms_dict.get('overall_rms', float('inf'))
            if overall_rms > overall_rms_threshold:
                skipped_by_rms += 1
                continue

            # 读取预测结果
            pred_path = seed_folder / 'all_predictions.csv'
            if not pred_path.exists():
                pred_path = seed_folder / 'predictions.csv'
            if not pred_path.exists():
                print(f"[Warning] No prediction CSV for seed {random_seed}, skipping...")
                skipped_by_rms += 1
                continue

            pred_df = pd.read_csv(pred_path)
            results_dict = {'all_results': pred_df.to_dict(orient='list')}

            # 存储结果（已通过 overall_rms 筛选）
            all_results.append({
                'seed': random_seed,
                'output_dir': seed_folder,
                'results_dict': results_dict,
                'rms_dict': rms_dict,
                'train_rms': rms_dict.get('train_rms', float('inf')),
                'test_rms': rms_dict.get('test_rms', float('inf')),
                'overall_rms': overall_rms
            })

        print(f"[Mode] Loaded {len(all_results)} seeds (skipped {skipped_by_rms} by overall_rms > {overall_rms_threshold})")

    else:
        # ========== 完整模式：执行迁移学习 ==========
        print("\n[Mode] FULL - Running transfer learning...")

        seed_folders = sorted([d for d in output_base.iterdir()
                               if d.is_dir() and d.name.startswith('seed')])

        for seed_folder in seed_folders:
            try:
                random_seed = int(seed_folder.name.replace('seed', ''))
            except ValueError:
                continue

            transfer_output_dir = transfer_output_base / f'seed{random_seed}'
            transfer_output_dir.mkdir(parents=True, exist_ok=True)

            # 查找 Alpha 模型
            alpha_model_path = None
            patterns = alpha_model_patterns or ['decay_model_seed{seed}.pth']
            for pattern in patterns:
                candidate = seed_folder / pattern.format(seed=random_seed)
                if candidate.exists():
                    alpha_model_path = candidate
                    break
            if alpha_model_path is None:
                print(f"[Skip] No Alpha model for seed {random_seed}")
                continue

            # 执行迁移学习
            results_dict, rms_dict = transfer_learn_from_alpha_model(
                alpha_model_path=str(alpha_model_path),
                train_csv_path=str(train_csv_path),
                test_csv_path=str(test_csv_path),
                output_dir=str(transfer_output_dir),
                freeze_first=freeze_first,
                max_epochs=max_epochs,
                patience=patience,
                lambda_init=lambda_init,
            )

            overall_rms = rms_dict.get('overall_rms', float('inf'))
            if overall_rms > overall_rms_threshold:
                skipped_by_rms += 1
                continue

            # 存储结果（已通过 overall_rms 筛选）
            all_results.append({
                'seed': random_seed,
                'output_dir': transfer_output_dir,
                'results_dict': results_dict,
                'rms_dict': rms_dict,
                'train_rms': rms_dict.get('train_rms', float('inf')),
                'test_rms': rms_dict.get('test_rms', float('inf')),
                'overall_rms': overall_rms
            })

        print(f"[Mode] Completed transfer learning for {len(all_results)} seeds (skipped {skipped_by_rms} by overall_rms > {overall_rms_threshold})")

    # ========== 检查结果 ==========
    if len(all_results) == 0:
        print(f"[Error] No valid results found after overall_rms filtering (threshold={overall_rms_threshold})")
        return None

    actual_k = min(top_k_models, len(all_results))
    print(f"\n{'=' * 80}")
    print(f"Total seeds after filtering: {len(all_results)}, selecting Top-{actual_k} for each path")
    print(f"{'=' * 80}")

    # # ========== 路径 A: 基于模型参数距离筛选 ==========
    # print("\n[Path A] Parameter-based selection...")
    # model_selected_seeds = select_best_by_parameter_distance(
    #     transfer_output_base=transfer_output_base,
    #     all_results=all_results,
    #     top_k=actual_k,
    #     model_pattern='model_best.pth' if not skip_transfer else 'transfer_model.pth'
    # )
    # _save_final_csv(
    #     all_results=all_results,
    #     selected_seeds=model_selected_seeds,
    #     output_path=transfer_output_base / 'transfer_seeds_model.csv',
    #     group_size=group_size
    # )
    # _print_seed_summary(all_results, model_selected_seeds, "Path A")

    # ========== 路径 B: 基于紧凑簇的 ratio 相似性筛选 ==========
    print("\n[Path B] Compact cluster selection...")
    mad_selected_seeds20 = select_best_by_mad_similarity(
        all_results=all_results,
        top_k=20,
        ratio_col='ratio',
    )
    _save_final_csv(
        all_results=all_results,
        selected_seeds=mad_selected_seeds20,
        output_path=transfer_output_base / 'transfer_seeds_ratio20.csv',
        group_size=group_size
    )
    _print_seed_summary(all_results, mad_selected_seeds20, "Path B 20")

    print("\n[Path B] Compact cluster selection...")
    mad_selected_seeds30 = select_best_by_mad_similarity(
        all_results=all_results,
        top_k=30,
        ratio_col='ratio',
    )
    _save_final_csv(
        all_results=all_results,
        selected_seeds=mad_selected_seeds30,
        output_path=transfer_output_base / 'transfer_seeds_ratio30.csv',
        group_size=group_size
    )
    _print_seed_summary(all_results, mad_selected_seeds30, "Path B 30")

    print("\n[Path B] Compact cluster selection...")
    mad_selected_seeds40 = select_best_by_mad_similarity(
        all_results=all_results,
        top_k=40,
        ratio_col='ratio',
    )
    _save_final_csv(
        all_results=all_results,
        selected_seeds=mad_selected_seeds40,
        output_path=transfer_output_base / 'transfer_seeds_ratio40.csv',
        group_size=group_size
    )
    _print_seed_summary(all_results, mad_selected_seeds40, "Path B 40")

    print("\n[Path B] Compact cluster selection...")
    mad_selected_seeds50 = select_best_by_mad_similarity(
        all_results=all_results,
        top_k=50,
        ratio_col='ratio',
    )
    _save_final_csv(
        all_results=all_results,
        selected_seeds=mad_selected_seeds50,
        output_path=transfer_output_base / 'transfer_seeds_ratio50.csv',
        group_size=group_size
    )
    _print_seed_summary(all_results, mad_selected_seeds50, "Path B 50")


    # # ========== 路径 C: 基于测试核 ratio 均值距 0 距离筛选 ==========
    # print("\n[Path C] Ratio mean distance to zero selection...")
    # mean_selected_seeds = select_best_by_ratio_mean_distance(
    #     all_results=all_results,
    #     top_k=actual_k,
    #     ratio_col='ratio'
    # )
    # _save_final_csv(
    #     all_results=all_results,
    #     selected_seeds=mean_selected_seeds,
    #     output_path=transfer_output_base / 'transfer_seeds_ratio_mean.csv',
    #     group_size=group_size
    # )
    # _print_seed_summary(all_results, mean_selected_seeds, "Path C")
    #
    # # ========== 路径 D: 基于层次聚类筛选 ==========
    # print("\n[Path D] Hierarchical clustering selection...")
    # cluster_selected_seeds = select_best_by_hierarchical_clustering(
    #     all_results=all_results,
    #     group_size=actual_k,
    #     ratio_col='ratio'
    # )
    # _save_final_csv(
    #     all_results=all_results,
    #     selected_seeds=cluster_selected_seeds,
    #     output_path=transfer_output_base / 'transfer_seeds_Kmean.csv',
    #     group_size=group_size
    # )
    # _print_seed_summary(all_results, cluster_selected_seeds, "Path D")

    print(f"\n{'=' * 80}")
    print("✓ Batch processing completed. Four outputs generated.")
    print(f"{'=' * 80}")

    # return {
    #     'model_path': transfer_output_base / 'transfer_seeds_model.csv',
    #     'ratio_path': transfer_output_base / 'transfer_seeds_ratio.csv',
    #     'ratio_mean_path': transfer_output_base / 'transfer_seeds_ratio_mean.csv',
    #     'Kmean_path': transfer_output_base / 'transfer_seeds_Kmean.csv'
    # }

    return {
        'ratio_path': transfer_output_base / 'transfer_seeds_ratio.csv',
    }

