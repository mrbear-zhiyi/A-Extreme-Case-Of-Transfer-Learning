from utils.utils_DNN_Train import *
import config
import time

"""
训练函数集成调用函数
1、结合config调用训练函数   ----   根据项目相对路径自适应输出结构
2、集成调用训练函数         ----   简化形参配置及集成结果分析
"""



def train_alpha_models():

    # ===============================================训练函数初始配置模块===================================================
    all_layers = sorted(config.ALPHA_LAYER_CONFIGS.keys())
    print()


    # ==================================================训练函数模块======================================================
    print(f"==========================================TRAINING ALPHA DECAY MODELS==========================================")
    start_time = time.time()
    for layer_num in all_layers:
        for neurons in config.ALPHA_LAYER_CONFIGS[layer_num]:
            print(f"\nTraining {layer_num}-Hidden layer alpha model with {neurons} Hidden neurons")
            model_start_time = time.time()
            output_dir = str(config.get_alpha_output_dir(layer_num, neurons))
            os.makedirs(output_dir, exist_ok=True)

            train_and_evaluate_with_hparams(
                file_path=str(config.ALPHA_DATA_PATH),
                best_hparams=config.get_alpha_training_config(layer_num, neurons),
                output_dir=output_dir
            )
            model_time = time.time() - model_start_time
            print(f"Completed {layer_num}-layer {neurons}-neuron alpha model | Time: {model_time / 60:.2f} minutes")

    total_time = time.time() - start_time
    print(f"ALPHA MODEL TRAINING COMPLETED | Total time: {total_time / 60:.2f} minutes")
    print()


def train_cluster_models_TL():
    # ===============================================训练函数初始配置模块===================================================
    all_layers = sorted(config.CLUSTER_LAYER_CONFIGS.keys())
    print()

    # ==================================================训练函数模块======================================================
    print(f"==========================================TRAINING CLUSTER DECAY MODELS==========================================")
    start_time = time.time()
    for layer_num in all_layers:
        print(f"TRAINING {layer_num}-LAYER CLUSTER MODELS")
        for neurons in config.CLUSTER_LAYER_CONFIGS[layer_num]:
            print(f"\nTraining {layer_num}-Hidden layer cluster model with {neurons}Hidden neurons")
            model_start_time = time.time()
            output_dir = str(config.get_cluster_output_dir(layer_num, neurons)) + '_train10'
            os.makedirs(output_dir, exist_ok=True)

            train_and_evaluate_with_hparams_TL(
                file_path=str(config.CLUSTER_TRAIN_DATA_PATH),
                cluster_train_file=None,
                cluster_test_file=str(config.CLUSTER_TEST_DATA_PATH),
                best_hparams=config.get_combined_training_config(layer_num, neurons),
                output_dir=output_dir
            )


            model_time = time.time() - model_start_time
            print(f"Completed {layer_num}-layer {neurons}-neuron cluster model | Time: {model_time / 60:.2f} minutes")

    total_time = time.time() - start_time
    print(f"CLUSTER MODEL TRAINING COMPLETED | Total time: {total_time / 60:.2f} minutes")
    print()



def train_combined_models_TL():
    # ===============================================训练函数初始配置模块===================================================
    all_layers = sorted(config.COMBINED_LAYER_CONFIGS.keys())
    print()

    # ==================================================训练函数模块======================================================
    print(f"==========================================TRAINING COMBINED MODELS==========================================")
    start_time = time.time()

    for layer_num in all_layers:
        print(f"TRAINING {layer_num}-LAYER COMBINED MODELS")
        for neurons in config.COMBINED_LAYER_CONFIGS[layer_num]:
            print(f"\nTraining {layer_num}-Hidden layer combined model with {neurons}Hidden neurons")
            model_start_time = time.time()
            output_dir = str(config.get_combined_output_dir(layer_num, neurons)) + '_train10'
            os.makedirs(output_dir, exist_ok=True)

            train_and_evaluate_with_hparams_TL(
                file_path=str(config.ALPHA_DATA_PATH),
                cluster_train_file=str(config.CLUSTER_TRAIN_DATA_PATH),
                cluster_test_file=str(config.CLUSTER_TEST_DATA_PATH),
                best_hparams=config.get_combined_training_config(layer_num, neurons),
                output_dir=output_dir
            )
            model_time = time.time() - model_start_time
            print(f"Completed {layer_num}-layer {neurons}-neuron combined model | Time: {model_time / 60:.2f} minutes")

    total_time = time.time() - start_time
    print(f"COMBINED MODEL TRAINING COMPLETED | Total time: {total_time / 60:.2f} minutes")
    print()


def evaluate_TL_models():
    """
    Evaluate transfer learning model performance (dual-module design)
    Module 1: Generate CSV files (evaluation loop)
    Module 2: Generate wide-format RMS summary from CSV files (2 rows × 200 columns)

    Usage:
      - Full run: Keep both modules enabled
      - CSV generation only: Set RUN_MODULE_1=True, RUN_MODULE_2=False
      - Summary generation only: Set RUN_MODULE_1=False, RUN_MODULE_2=True (requires existing CSV files)
    """
    import time

    tl_results_dir = config.TRANSFER_RESULTS_DIR
    total_configs = len(config.EVALUATE_LAYER_CONFIGS_TL)

    # ========================================================================
    # Module 1: Evaluation loop - Generate all cluster_extrapolation_results.csv files
    # ========================================================================
    RUN_MODULE_1 = True

    if RUN_MODULE_1:
        start_time = time.time()
        print(f"Module 1 started at {time.strftime('%H:%M:%S')}")

        results_list = []
        successful_configs = 0

        for i, config_item in enumerate(config.EVALUATE_LAYER_CONFIGS_TL, 1):
            (layer, neurons_val), (train_num, sort_num) = config_item
            best_hparams = config.get_combined_training_config(layer, neurons_val)

            # 初始化 mode_results 字典（修复：之前始终为空）
            mode_results = {
                'cluster_only': {},
                'combined': {}
            }

            train_csv_path = tl_results_dir / f"model_{layer}_{neurons_val}" / f"train_k{train_num}_row{sort_num}.csv"
            test_csv_path = tl_results_dir / f"model_{layer}_{neurons_val}" / f"test_k{train_num}_row{sort_num}.csv"

            if not train_csv_path.exists() or not test_csv_path.exists():
                continue

            try:
                output_dir = tl_results_dir / f"model_{layer}_{neurons_val}_k{train_num}_row{sort_num}"
                result_csv_path1 = output_dir / "extrapolation_results.csv"
                result_csv_path2 = output_dir / "K-fold_results.csv"

                # 生成绘图专用CSV（含logT_pred列）
                cluster_results, combined_results = evaluate_single_subset_logscale_output_corrected(
                    train_csv_path=str(train_csv_path),
                    test_csv_path=str(test_csv_path),
                    alpha_data_path=str(config.ALPHA_DATA_PATH),
                    output_base_path=str(result_csv_path1),
                )

                # 生成10折CV结果
                evaluate_single_subset_10fold_rms_output_corrected(
                    train_csv_path=str(train_csv_path),
                    test_csv_path=str(test_csv_path),
                    alpha_data_path=str(config.ALPHA_DATA_PATH),
                    output_base_path=str(result_csv_path2),
                    optimizer_type='LM',
                )

                # 验证文件生成
                if not (output_dir / "extrapolation_results_cluster_logscale.csv").exists():
                    raise FileNotFoundError(f"Cluster CSV not generated in {output_dir}")
                if not (output_dir / "extrapolation_results_combined_logscale.csv").exists():
                    raise FileNotFoundError(f"Combined CSV not generated in {output_dir}")

                # 从返回的DataFrame计算RMS并填充mode_results（修复：之前未填充）
                if len(cluster_results) > 0:
                    co_rms = np.sqrt(np.mean((cluster_results['logT_exp'] - cluster_results['logT_pred']) ** 2))
                    mode_results['cluster_only']['total_rms'] = co_rms
                if len(combined_results) > 0:
                    comb_rms = np.sqrt(np.mean((combined_results['logT_exp'] - combined_results['logT_pred']) ** 2))
                    mode_results['combined']['total_rms'] = comb_rms

            except Exception as e:
                print(f"  [Warning] Config {i}/{total_configs} failed: {str(e)}")
                continue

            combination_name = f"{layer}L_{neurons_val}N_k{train_num}_row{sort_num}"
            result_row = {
                'combination_name': combination_name,
                'layer': layer,
                'neurons': neurons_val,
                'train_samples': train_num,
                'sort_row': sort_num,
            }

            for mode in ["combined", "cluster_only"]:
                if mode in mode_results and 'total_rms' in mode_results[mode]:
                    result_row[f'{mode}_total_rms'] = mode_results[mode]['total_rms']
                    result_row[f'{mode}_train_rms'] = None  # 10-fold CV中可补充
                    result_row[f'{mode}_test_rms'] = None
                    result_row[f'{mode}_train_count'] = len(cluster_results) if mode == 'cluster_only' else len(
                        combined_results)
                    result_row[f'{mode}_test_count'] = 0
                else:
                    result_row[f'{mode}_total_rms'] = None
                    result_row[f'{mode}_train_rms'] = None
                    result_row[f'{mode}_test_rms'] = None
                    result_row[f'{mode}_train_count'] = None
                    result_row[f'{mode}_test_count'] = None

            results_list.append(result_row)
            successful_configs += 1

            if successful_configs % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {successful_configs}/{total_configs} configurations completed ({elapsed:.1f}s)")

        if results_list:
            final_results_df = pd.DataFrame(results_list)
            final_results_path = tl_results_dir / 'exploration.csv'
            final_results_df.to_csv(str(final_results_path), index=False)
            module1_time = time.time() - start_time
            print(
                f"Module 1 completed in {module1_time:.2f} seconds | Successful configs: {len(results_list)}/{total_configs}")
        else:
            module1_time = time.time() - start_time
            print(f"Module 1 completed with NO valid results in {module1_time:.2f} seconds (results_list empty)")
            final_results_df = None  # 不返回，继续执行Module 2
    else:
        final_results_df = None

    # ========================================================================
    # Module 2: CSV aggregation - Generate wide-format RMS summary (2 rows × 200 columns)
    # ========================================================================
    RUN_MODULE_2 = False

    if RUN_MODULE_2:
        start_time = time.time()
        expected_configs = 200
        wide_df = pd.DataFrame(
            index=['cluster_only', 'combined'],
            columns=range(expected_configs),
            dtype=float
        )

        filled = {'cluster_only': 0, 'combined': 0}

        for config_item in config.EVALUATE_LAYER_CONFIGS_TL:
            (layer, neurons_val), (train_num, sort_num) = config_item

            try:
                sort_num = int(sort_num)
            except (TypeError, ValueError):
                print(f"Warning: sort_num '{sort_num}' invalid (non-integer), skipping config: {config_item}")
                continue
            if not (0 <= sort_num < expected_configs):
                continue

            # 构造统一的结果目录路径
            result_dir = tl_results_dir / f"model_{layer}_{neurons_val}_k{train_num}_row{sort_num}"

            file_map = {
                'cluster_only': result_dir / "extrapolation_results_cluster_logscale_AdamW.csv",
                'combined': result_dir / "extrapolation_results_combined_logscale_AdamW.csv"
            }

            for mode, csv_path in file_map.items():
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if 'logT_exp' in df.columns and 'logT_pred' in df.columns:
                            residuals = df['logT_exp'] - df['logT_pred']
                            valid_residuals = residuals[np.isfinite(residuals)]
                            if len(valid_residuals) > 0:
                                total_rms = float(np.sqrt(np.mean(valid_residuals ** 2)))
                                wide_df.loc[mode, sort_num] = total_rms
                                filled[mode] += 1
                        else:
                            print(f"Warning: Missing required columns in {csv_path}. "
                                  f"Available: {list(df.columns)}. Skipping.")
                    except Exception as e:
                        print(f"Warning: Error processing {csv_path}: {str(e)}")
                        continue

        # 保存宽格式汇总结果
        wide_results_path = tl_results_dir / 'rms_summary_wide.csv'
        wide_df.to_csv(str(wide_results_path), index=True)
        module2_time = time.time() - start_time

        # 打印统计信息
        print(f"\nModule 2 completed in {module2_time:.2f} seconds")
        print(f"   rms_summary_wide.csv saved to: {wide_results_path}")
        print(f"   Filled entries - cluster_only: {filled['cluster_only']}/200, combined: {filled['combined']}/200")
    else:
        wide_df = None

    if RUN_MODULE_1 and RUN_MODULE_2:
        return final_results_df, wide_df
    elif RUN_MODULE_1:
        return final_results_df
    elif RUN_MODULE_2:
        return wide_df
    else:
        return None


def Seed_evaluation_TL(
        seed_list: list = None,  # 【修改点】直接传入种子列表
        best_hparams: dict = None,
):
    """批量种子评估调用函数（支持自定义种子列表）"""

    # ========== 配置初始化 ==========
    # 如果未传入种子列表，默认使用原有的生成逻辑，保证兼容性
    if seed_list is None:
        seed_list = [9508 + i * 1 for i in range(1000)]

    # 确保种子列表有效
    if not seed_list or not isinstance(seed_list, (list, tuple)):
        raise ValueError("seed_list 必须是一个非空的整数列表，例如 [42, 62, 82]")

    if best_hparams is None:
        all_layers = sorted(config.EVALUATE_LAYER_CONFIGS.keys())
        best_hparams = config.get_combined_training_config(
            layer=all_layers[0],
            neurons=config.EVALUATE_LAYER_CONFIGS[all_layers[0]][0]
        )

    # ========== 文件检查 ==========
    for path in [config.ALPHA_DATA_PATH, config.CLUSTER_TRAIN_DATA_PATH,
                 config.CLUSTER_TEST_DATA_PATH, config.CLUSTER_DATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    # 初始化汇总容器
    all_alpha_list, all_cluster_list, all_metrics_list = [], [], []
    output_base = Path(config.MULTI_SEED_OUTPUT_DIR)
    output_base.mkdir(parents=True, exist_ok=True)

    # ========== 主循环：逐种子评估 ==========
    # 【修改点】直接遍历传入的 seed_list
    for random_seed in seed_list:
        output_dir = str(output_base / f'seed{random_seed}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"[Running] Seed: {random_seed} ...")

        # 调用评估函数
        evaluate_Seed_cluster(
            alpha_file_path=str(config.ALPHA_DATA_PATH),
            best_hparams=best_hparams,
            train_csv_path=str(config.CLUSTER_TRAIN_DATA_PATH),
            test_csv_path=str(config.CLUSTER_TEST_DATA_PATH),
            output_dir=output_dir,
            random_seed=random_seed,
            use_global=False
        )
        # evaluate_Seed_alpha(
        #     alpha_file_path=str(config.ALPHA_DATA_PATH),
        #     best_hparams=best_hparams,
        #     output_dir=output_dir,
        #     seed=random_seed
        # )

        # 收集该种子的结果文件
        alpha_path = Path(output_dir) / 'results_alpha.csv'
        cluster_path = Path(output_dir) / 'results_cluster.csv'
        metrics_path = Path(output_dir) / 'metrics_summary.csv'

        if alpha_path.exists():
            df_alpha = pd.read_csv(str(alpha_path))
            df_alpha['seed'] = random_seed  # 添加种子标识
            all_alpha_list.append(df_alpha)

        if cluster_path.exists():
            df_cluster = pd.read_csv(str(cluster_path))
            df_cluster['seed'] = random_seed  # 添加种子标识
            all_cluster_list.append(df_cluster)

        if metrics_path.exists():
            all_metrics_list.append(pd.read_csv(str(metrics_path)))

    # ========== 汇总保存 ==========
    # 1. 汇总所有 alpha 结果
    if all_alpha_list:
        all_alpha_df = pd.concat(all_alpha_list, ignore_index=True)
        all_alpha_df.to_csv(output_base / 'results_alpha.csv', index=False)
        print(f"[Done] Alpha results: {len(all_alpha_df)} rows → {output_base / 'results_alpha.csv'}")

    # 2. 汇总所有 cluster 结果
    if all_cluster_list:
        all_cluster_df = pd.concat(all_cluster_list, ignore_index=True)
        all_cluster_df.to_csv(output_base / 'results_cluster.csv', index=False)
        print(f"[Done] Cluster results: {len(all_cluster_df)} rows → {output_base / 'results_cluster.csv'}")

    # 3. 汇总所有种子指标
    if all_metrics_list:
        all_metrics_df = pd.concat(all_metrics_list, ignore_index=True)
        all_metrics_df.to_csv(output_base / 'metrics_all_seeds.csv', index=False)
        print(f"[Done] Metrics summary: {len(all_metrics_df)} rows → {output_base / 'metrics_all_seeds.csv'}")

    return all_alpha_list, all_cluster_list, all_metrics_list

