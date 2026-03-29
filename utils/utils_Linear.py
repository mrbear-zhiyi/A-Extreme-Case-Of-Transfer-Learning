import itertools
import heapq
import config
import math
import time
import pandas as pd
from scipy.optimize import curve_fit
import os
import numpy as np
from pathlib import Path
"""
线性处理模块

====功能模块====
1、数据检验函数
2、迁移簇类数据索引函数
3、迁移UDL对比函数

====辅助模块====
1、UDL拟合函数
2、UDL外推函数
3、RMS计算函数
"""



"""
================================================功能模块================================================
"""
def main_fit_all(alpha_csv_path: str, cluster_csv_path: str, output_dir: str):

    # ============================================== 数据读取 ==================================================
    df_alpha = pd.read_csv(alpha_csv_path)
    df_cluster = pd.read_csv(cluster_csv_path)
    df_combined = pd.concat([df_alpha, df_cluster], ignore_index=True)
    # 标准参数
    std_params = {
        "alpha-only": np.array([0.4065, -0.4311, -20.7889]),
        "cluster-only": np.array([0.3671, -0.3296, -26.2681]),
        "alpha+cluster": np.array([0.3949, -0.3693, -23.7615])
    }

    # ============================================== 数据拟合 ==================================================
    fit_results = []

    def fit_and_calc(df, decay_type):
        try:
            if df.empty:
                return [decay_type, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            a, b, c, rmse = udl_fit_core(df, decay_type=decay_type)
            l2 = np.linalg.norm(np.array([a, b, c]) - std_params[decay_type])
            return [decay_type, a, b, c, rmse, std_params[decay_type][0], std_params[decay_type][1],
                    std_params[decay_type][2], l2]
        except Exception as e:
            print(f"Fitting fail:({decay_type}): {str(e)}")
            return [decay_type, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    fit_results.append(fit_and_calc(df_alpha, "alpha-only"))
    fit_results.append(fit_and_calc(df_cluster, "cluster-only"))
    fit_results.append(fit_and_calc(df_combined, "alpha+cluster"))

    # ============================================== 保存结果 ==================================================
    columns = ['decay_type', 'a', 'b', 'c', 'RMSE', 'a_std', 'b_std', 'c_std', 'L2_distance_to_std']
    summary_df = pd.DataFrame(fit_results, columns=columns)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "udl_fit_alpha_cluster.csv"
    summary_df.to_csv(output_path, index=False)

    print(f"Fitting done. Result saved to: {output_path}")
    return summary_df


def cluster_subsets(alpha_csv_path: str, cluster_csv_path: str) -> dict:
    # ============================================== 数据加载 ==================================================
    import pandas as pd
    import numpy as np
    import math
    import itertools
    import heapq
    import time
    import config  # 新增：导入config模块

    df_a = pd.read_csv(alpha_csv_path)
    df_c = pd.read_csv(cluster_csv_path)
    n_cluster = len(df_c)
    print(f"Total cluster nuclei: {n_cluster}")

    # 从config读取子集数量（关键修改点）
    top_k_global = getattr(config, 'CLUSTER_INDICES', 200)
    print(f"Using top {top_k_global} subsets per k-value (from config.CLUSTER_INDICES)")

    sizes = (3, 4, 5, 6, 7, 8, 9, 10)
    std = np.array([0.3949, -0.3693, -23.7615])
    topk_data = {}

    # ============================================== 数据处理 ==================================================
    for k in sizes:
        top_k = top_k_global  # 使用配置值

        if k > n_cluster:
            print(f"Warning: k={k} exceeds total cluster nuclei ({n_cluster}). Skipping.")
            continue

        total_combinations = math.comb(n_cluster, k)
        print(f"\nStarting k={k}: Processing all {total_combinations:,} combinations...")

        heap = []
        start_time = time.time()
        processed_count = 0
        valid_count = 0

        for idx_tuple in itertools.combinations(range(n_cluster), k):
            processed_count += 1
            if processed_count % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  k={k}: Processed {processed_count:,}/{total_combinations:,} combinations "
                      f"({processed_count / total_combinations * 100:.1f}%, "
                      f"{elapsed:.1f}s elapsed, {valid_count} valid)")

            df_fit = pd.concat([df_a, df_c.iloc[list(idx_tuple)]], ignore_index=True)
            try:
                a, b, c, _ = udl_fit_core(df_fit)
            except Exception as e:
                continue

            if np.any(np.isnan([a, b, c])):
                continue

            l2 = np.linalg.norm([a, b, c] - std)
            iso = df_c.iloc[list(idx_tuple)]['Isotope'].tolist() if 'Isotope' in df_c else [f"idx_{i}" for i in
                                                                                            idx_tuple]
            valid_count += 1

            if len(heap) < top_k:
                heapq.heappush(heap, (-l2, a, b, c, iso, idx_tuple))
            elif l2 < -heap[0][0]:
                heapq.heapreplace(heap, (-l2, a, b, c, iso, idx_tuple))

        results = []
        while heap:
            neg_l2, a, b, c, iso, idx = heapq.heappop(heap)
            results.append((-neg_l2, a, b, c, iso, list(idx)))
        results.sort()

        if not results:
            print(f"No valid results found for k={k}. Skipping.")
            continue

        # ============================================== 数据导出 ==================================================
        k_results = []
        for rank, (l2_dist, a, b, c, iso, indices) in enumerate(results, 1):
            k_results.append({
                'Rank': rank,
                'L2_Distance': l2_dist,
                'a': a,
                'b': b,
                'c': c,
                'Cluster_Isotopes': '; '.join(iso),
                'Cluster_Indices': indices
            })

        topk_data[k] = k_results
        elapsed_time = time.time() - start_time
        print(f"Completed k={k}: Found {len(results)}/{valid_count} valid results. (Time: {elapsed_time:.2f}s)")
    print(f"{'=' * 60}")

    return topk_data


def udl_rms_alpha_cluster(
        alpha_csv_path: str,
        cluster_csv_path: str,
        output_dir: str = str(config.LINEAR_DIR)
):
    """
    计算全局UDL参数和RMS值，使用所有alpha数据 + 所有cluster数据
    返回单个UDL值而不是多个组合的值
    """
    # ============================================== 数据加载 ==================================================
    os.makedirs(output_dir, exist_ok=True)
    df_alpha = pd.read_csv(alpha_csv_path)
    df_cluster = pd.read_csv(cluster_csv_path)

    # ============================================== 全局UDL拟合 ==================================================
    # 使用所有alpha数据 + 所有cluster数据进行拟合
    df_combined = pd.concat([df_alpha, df_cluster], ignore_index=True)
    a, b, c, _ = udl_fit_core(df_combined)

    if np.any(np.isnan([a, b, c])):
        raise ValueError("UDL fitting failed - got NaN parameters")

    # 计算cluster数据上的预测和RMS
    logT_cluster_true = np.log10(df_cluster['half_life_s'].values.astype(np.float32))
    logT_pred = udl_predict_core(df_cluster, a, b, c)
    overall_rms = calculate_rms(logT_cluster_true, logT_pred)

    # STD参数用于参考
    STD_PARAMS = np.array([0.3949, -0.3693, -23.7615])
    l2_dist = np.linalg.norm(np.array([a, b, c]) - STD_PARAMS)

    print(f"\n" + "=" * 60)
    print("GLOBAL UDL FITTING RESULTS (All Alpha + All Cluster Data)")
    print(f"Fitted parameters: a={a:.6f}, b={b:.6f}, c={c:.6f}")
    print(f"Overall RMS on cluster data: {overall_rms:.6f}")
    print(f"L2 distance to standard parameters: {l2_dist:.6f}")
    print("=" * 60 + "\n")

    # ============================================== 保存单个UDL值 ================================================
    udl_record = {
        'model_type': 'global_udl_all_data',
        'a': a,
        'b': b,
        'c': c,
        'overall_sigma_rms': overall_rms,
        'param_l2_distance_to_std': l2_dist,
        'total_alpha_nuclei': len(df_alpha),
        'total_cluster_nuclei': len(df_cluster),
        'fitting_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    udl_df = pd.DataFrame([udl_record])
    output_path = config.UDL_CSV_PATH
    udl_df.to_csv(output_path, index=False)
    print(f"Global UDL results saved to: {output_path}")

    return udl_df

def generate_udl_extrapolation(
        alpha_csv_path: str,
        cluster_csv_path: str,
        exploration_csv_path: str,
        debug_output_path: str = "./results/03_visualization/isotope_scatter_vertical/udl_extrapolation.csv"
) -> pd.DataFrame:
    # ==================== 1. 加载并验证训练数据 ====================
    df_alpha = pd.read_csv(alpha_csv_path)
    df_cluster = pd.read_csv(cluster_csv_path)

    # 严格验证必要列
    required_cols = ['Z', 'A', 'Q_MeV', 'half_life_s', 'Z_k', 'A_k']
    for df, name in [(df_alpha, 'alpha'), (df_cluster, 'cluster')]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{name} data missing required columns: {missing}")

    # 合并前排除114Ba（拟合阶段）
    mask_114ba_alpha = (df_alpha['Z'] == 56) & (df_alpha['A'] == 114)
    mask_114ba_cluster = (df_cluster['Z'] == 56) & (df_cluster['A'] == 114)
    if mask_114ba_alpha.any():
        df_alpha = df_alpha[~mask_114ba_alpha].reset_index(drop=True)
    if mask_114ba_cluster.any():
        df_cluster = df_cluster[~mask_114ba_cluster].reset_index(drop=True)

    df_combined = pd.concat([df_alpha, df_cluster], ignore_index=True)

    # ==================== 2. UDL参数拟合 ====================
    try:
        a, b, c, rmse_fit = udl_fit_core(df_combined)
        if np.isnan(a) or np.isnan(b) or np.isnan(c):
            raise RuntimeError("UDL fitting returned NaN parameters")
    except Exception as e:
        raise RuntimeError(f"UDL fitting failed: {e}")

    # ==================== 3. 加载外推数据并严格验证 ====================
    df_exploration = pd.read_csv(exploration_csv_path)

    # 验证外推数据列完整性
    missing_expl = [col for col in required_cols if col not in df_exploration.columns]
    if missing_expl:
        raise ValueError(f"Exploration data missing columns: {missing_expl}")

    # 严格排除114Ba（外推阶段）
    mask_114ba_expl = (df_exploration['Z'] == 56) & (df_exploration['A'] == 114)
    if mask_114ba_expl.any():
        df_exploration = df_exploration[~mask_114ba_expl].reset_index(drop=True)

    # ==================== 4. UDL预测 ====================
    try:
        logT_pred_udl = udl_predict_core(df_exploration, a, b, c)
    except Exception as e:
        raise RuntimeError(f"UDL prediction failed: {e}")

    # ==================== 5. 构建核素标签和唯一ID ====================
    element_symbols = {
        56: 'Ba', 84: 'Po', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
        93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
        101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf'
    }

    nucleus_labels = []
    nuclide_ids = []
    for z, a_val in zip(df_exploration['Z'], df_exploration['A']):
        symbol = element_symbols.get(int(z), '?')
        nucleus_labels.append(f"$^{{{int(a_val)}}}${symbol}")
        nuclide_ids.append(f"{int(z)}-{int(a_val)}")

    # ==================== 6. 构建结果DataFrame ====================
    logT_exp = np.log10(df_exploration['half_life_s'].values.astype(np.float64))

    results_df = pd.DataFrame({
        'nuclide_id': nuclide_ids,
        'nucleus_label': nucleus_labels,
        'Z': df_exploration['Z'].values,
        'A': df_exploration['A'].values,
        'Z_k': df_exploration['Z_k'].values,
        'A_k': df_exploration['A_k'].values,
        'Q_MeV': df_exploration['Q_MeV'].values,
        'Emitted_Particle': df_exploration.get('Emitted_Particle', pd.Series(dtype=str)),
        'half_life_s_exp': df_exploration['half_life_s'].values,
        'logT_exp': logT_exp,
        'logT_pred_udl': logT_pred_udl,
        'certainty': df_exploration.get('certainty', pd.Series(True, index=df_exploration.index))
    })

    # ==================== 7. 调试输出（无日志） ====================
    Path(debug_output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(debug_output_path, index=False)

    return results_df


"""
================================================辅助模块================================================
"""
def udl_fit_core(df: pd.DataFrame, decay_type: str = "unknown", return_details: bool = False):

    # ============================================== 数据提取 ==================================================
    df = df.copy()
    required_cols = ['Z', 'A', 'Q_MeV', 'half_life_s', 'Z_k', 'A_k']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing:{missing}(CSV need Z_k/A_k column)")

    Z = df['Z'].to_numpy()
    A = df['A'].to_numpy()
    Q = df['Q_MeV'].to_numpy()
    T = df['half_life_s'].to_numpy()
    Z_k = df['Z_k'].to_numpy()
    A_k = df['A_k'].to_numpy()

    if len(df) < 3 or np.any(T <= 0) or np.any(Q <= 0) or not np.all(np.isfinite(T)):
        if return_details:
            return df.iloc[0:0], (np.nan, np.nan, np.nan, np.nan)
        return (np.nan, np.nan, np.nan, np.nan)

    # ============================================== UDL参数设定 ==================================================
    log10_T = np.log10(T)
    Z_d = Z - Z_k
    A_d = A - A_k
    mu = (A_k * A_d) / A
    chi_prime = Z_k * Z_d * np.sqrt(mu / Q)
    rho_prime = np.sqrt((mu * Z_k * Z_d) * (A_k ** (1 / 3) + A_d ** (1 / 3)))
    X = np.column_stack((chi_prime, rho_prime))

    # ============================================== 线性拟合 ==================================================
    def linear_model(x, a, b, c):
        return a * x[:, 0] + b * x[:, 1] + c

    try:
        params, _ = curve_fit(linear_model, X, log10_T, maxfev=10000)
    except Exception as e:
        print(f"Fitting fail({decay_type}): {e}")
        if return_details:
            return df.iloc[0:0], (np.nan, np.nan, np.nan, np.nan)
        return (np.nan, np.nan, np.nan, np.nan)

    a, b, c = params
    log10_T_cal = linear_model(X, a, b, c)
    residual = log10_T - log10_T_cal
    rmse = np.sqrt(np.mean(residual ** 2))

    # ============================================== 返回结果 ==================================================
    if return_details:
        df.loc[:, 'log10_T_exp'] = log10_T
        df.loc[:, 'log10_T_cal'] = log10_T_cal
        df.loc[:, 'residual'] = residual
        print(f"\nFitting for {decay_type}")
        print(f"  a = {a:.6f}, b = {b:.6f}, c = {c:.6f}, RMSE = {rmse:.6f}")
        return df, (a, b, c, rmse)
    return (a, b, c, rmse)


def udl_predict_core(df: pd.DataFrame, a: float, b: float, c: float) -> np.ndarray:
    Z = df['Z'].values
    A = df['A'].values
    Q = df['Q_MeV'].values
    Z_k = df['Z_k'].values
    A_k = df['A_k'].values

    Z_d = Z - Z_k
    A_d = A - A_k
    mu = (A_k * A_d) / A
    chi = Z_k * Z_d * np.sqrt(mu / Q)
    rho = np.sqrt((mu * Z_k * Z_d) * (A_k ** (1 / 3) + A_d ** (1 / 3)))
    logT_pred = a * chi + b * rho + c
    return logT_pred


def calculate_rms(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    return np.sqrt(np.mean((pred_values - true_values) ** 2))