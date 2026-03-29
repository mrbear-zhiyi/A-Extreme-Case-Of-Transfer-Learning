import json
import pandas as pd
import config
from typing import Dict, Any, List, Tuple, Optional
import os
import numpy as np
"""
数据预处理模块

=====功能模块=====
1、读取alpha函数
2、读取cluster函数
3、加载数据函数
"""



"""
================================================功能模块================================================
"""
def load_alpha_data(json_path: str) -> pd.DataFrame:

    #==============================================提取与统一单位辅助函数==================================================
    def _extract_value_unit(entry: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
        return (entry.get("value"), entry.get("unit")) if entry else (None, None)
    def _extract_value_unit_type(entry: Optional[Dict[str, Any]]) -> Tuple[
        Optional[float], Optional[str], Optional[str]]:
        if not entry:
            return None, None, None
        unc = entry.get("uncertainty")
        return entry.get("value"), entry.get("unit"), unc.get("type") if unc else None


    # =============================================json文件解析与数据提取=================================================
    with open(json_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)
    columns = [
        'Z', 'N', 'A', 'half_life', 'half_life_unit', 'half_life_type',
        'br_alpha', 'br_alpha_type', 'half_life_s', 'half_life_s_unit', 'Q_MeV', 'Q_MeV_unit'
    ]
    table = []
    rows = list(data_json.keys())

    for nuclide_name in rows:
        nuclide = data_json[nuclide_name]
        record = {k: None for k in columns}
        #基本信息解析
        record["Z"] = nuclide.get("z")
        record["N"] = nuclide.get("n")
        record["A"] = nuclide.get("a")
        record["Q_MeV"], record["Q_MeV_unit"] = _extract_value_unit(nuclide.get("alpha"))
        # 半衰期解析
        levels = nuclide.get("levels", [])
        level0 = next((lv for lv in levels if lv.get("energy", {}).get("value") == 0), levels[0] if levels else None)
        if level0 and "halflife" in level0:
            record['half_life'], record['half_life_unit'], record['half_life_type'] = _extract_value_unit_type(
                level0['halflife'])
        # 分支比解析
        observed = level0.get("decayModes", {}).get("observed", []) if level0 else []
        br_total = 0.0
        br_type = None
        for mode_info in observed:
            if mode_info.get("mode") == "A":
                br_val = mode_info.get("value")
                if br_val:
                    br_total += br_val
                    br_type = mode_info.get("uncertainty", {}).get("type")
        record['br_alpha'] = br_total
        record['br_alpha_type'] = br_type

        # =============================================数据后处理=================================================
        #半衰期单位统一
        total_T = record["half_life"]
        if total_T and record['half_life_unit'] and br_total:
            time_factor = {
                "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12,
                "m": 60.0, "h": 3600.0, "d": 86400.0, "y": 31557600.0
            }
            factor = time_factor.get(record['half_life_unit'], 1.0)
            record['half_life_s'] = total_T / (br_total / 100.0) * factor
            record['half_life_s_unit'] = "s"
        table.append([record[col] for col in columns])
    # 能量单位统一
    df = pd.DataFrame(table, index=rows, columns=columns)
    energy_factor = {"keV": 1e-3, "MeV": 1, "eV": 1e-6, "GeV": 1e3}
    df['Q_MeV'] = df.apply(
        lambda x: x['Q_MeV'] * energy_factor.get(x['Q_MeV_unit'], 1.0) if x['Q_MeV'] else x['Q_MeV'], axis=1
    )
    df['Q_MeV_unit'] = "MeV"
    # 筛选有效数据
    df = df.dropna(subset=['Q_MeV', 'half_life_s'])
    df = df[df['Q_MeV'] >= 0]
    if 'br_alpha_type' in df.columns:
        df = df[~df['br_alpha_type'].isin(["unreported", "limit"])]
    if 'half_life_type' in df.columns:
        df = df[~df['half_life_type'].isin(["unreported", "limit"])]

    # =============================================结果输出=================================================
    final_df = pd.DataFrame({
        'Z': df['Z'].astype(float),
        'N': df['N'].astype(float),
        'A': df['A'].astype(float),
        'Q_MeV': df['Q_MeV'].astype(float),
        'half_life_s': df['half_life_s'].astype(float),
        'br_alpha': df['br_alpha'].astype(float),
        'Emitted_Particle': ['alpha'] * len(df)
    })
    final_df['Z_k'] = final_df['Emitted_Particle'].apply(lambda x: config.PARTICLE_MAP.get(x.strip(), (2, 4))[0])
    final_df['A_k'] = final_df['Emitted_Particle'].apply(lambda x: config.PARTICLE_MAP.get(x.strip(), (2, 4))[1])

    print(f"Successfully extracted {len(final_df)} alpha decay nuclei from {os.path.basename(json_path)}")
    return final_df



def load_cluster_data() -> pd.DataFrame:
    data = [
        ["221Fr", 87, 134, 221, "14C", 31.29, 14.52, 8.14e-13],
        ["221Ra", 88, 133, 221, "14C", 32.394, 13.39, 1.15e-12],
        ["222Ra", 88, 134, 222, "14C", 33.049, 11.01, 3.7e-10],
        ["223Ra", 88, 135, 223, "14C", 31.829, 15.04, 8.9e-10],
        ["224Ra", 88, 136, 224, "14C", 30.535, 15.86, 4.3e-11],
        ["226Ra", 88, 138, 226, "14C", 28.196, 21.19, 3.2e-11],
        ["225Ac", 89, 136, 225, "14C", 30.476, 17.28, 4.5e-12],
        ["228Th", 90, 138, 228, "20O", 44.723, 20.72, 1.13e-13],
        ["230Th", 90, 140, 230, "24Ne", 57.758, 24.61, 5.6e-13],
        ["231Pa", 91, 140, 231, "23F", 51.844, 26.02, 9.97e-15],
        ["231Pa", 91, 140, 231, "24Ne", 60.408, 22.88, 1.34e-11],
        ["230U", 92, 138, 230, "22Ne", 61.388, 19.57, 4.8e-14],
        ["232U", 92, 140, 232, "24Ne", 62.309, 20.4, 9.16e-12],
        ["233U", 92, 141, 233, "24Ne", 60.484, 24.84, 7.2e-13],
        ["233U", 92, 141, 233, "25Ne", 60.776, 24.84, 7.2e-13],
        ["234U", 92, 142, 234, "28Mg", 74.108, 25.14, 1.38e-13],
        ["234U", 92, 142, 234, "24Ne", 58.825, 25.88, 9.9e-14],
        ["234U", 92, 142, 234, "26Ne", 59.465, 25.88, 9.9e-14],
        ["235U", 92, 143, 235, "24Ne", 57.361, 27.42, 8.06e-12],
        ["235U", 92, 143, 235, "25Ne", 57.756, 27.42, 8.06e-12],
        ["236U", 92, 144, 236, "28Mg", 70.560, 27.58, 2e-13],
        ["236U", 92, 144, 236, "30Mg", 72.299, 27.58, 2e-13],
        ["236Pu", 94, 142, 236, "28Mg", 79.668, 21.52, 2.7e-14],
        ["238Pu", 94, 144, 238, "32Si", 91.188, 25.27, 1.38e-16],
        ["238Pu", 94, 144, 238, "28Mg", 75.910, 25.7, 5.62e-17],
        ["238Pu", 94, 144, 238, "30Mg", 76.822, 25.7, 5.62e-17],
        ["242Cm", 96, 146, 242, "34Si", 96.508, 23.15, 1e-16]
    ]


    df = pd.DataFrame(data, columns=[
        'Isotope', 'Z', 'N', 'A', 'Emitted_Particle', 'Q_MeV', 'log_T_s', 'br_alpha'
    ])
    df['half_life_s'] = 10 ** df['log_T_s']
    df['Z_k'] = df['Emitted_Particle'].apply(lambda x: config.PARTICLE_MAP.get(x.strip(), (2, 4))[0])
    df['A_k'] = df['Emitted_Particle'].apply(lambda x: config.PARTICLE_MAP.get(x.strip(), (2, 4))[1])


    return df[['Z', 'N', 'A', 'Q_MeV', 'half_life_s', 'br_alpha', 'Emitted_Particle', 'Z_k', 'A_k']]


def load_data(file_path: str, test_size: float = 0.2, random_seed: int = 42,
              train_samples: int = None, n_splits: int = 1) -> dict:
    """
    加载数据，支持按比例划分、按指定数量划分，以及多组随机划分

    Args:
        file_path: CSV文件路径
        test_size: 测试集比例（当train_samples为None时使用）
        random_seed: 基础随机种子
        train_samples: 指定训练集样本数量（优先级高于test_size）
        n_splits: 要生成的随机划分组数（默认为1）

    Returns:
        当n_splits=1时：包含训练验证集、测试集和原始DataFrame的字典
        当n_splits>1时：包含n_splits个划分的列表，每个划分都是一个字典
    """

    # =============================================基础读取与校验=================================================
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path)
    required_columns = ['Z', 'N', 'A', 'Q_MeV', 'half_life_s', 'Emitted_Particle', 'Z_k', 'A_k']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # =============================================数据预处理=================================================
    df = df.dropna(subset=['half_life_s'])
    df['logT'] = np.log10(df['half_life_s'].astype(np.float32))
    features = df[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
    logT = df['logT'].values.astype(np.float32)
    N = df['N'].values.astype(np.float32)

    # =============================================划分数据集=================================================
    if n_splits == 1:
        # 单组划分（保持原有逻辑）
        np.random.seed(random_seed)
        total_samples = len(features)

        if train_samples is not None:
            if train_samples <= 0 or train_samples >= total_samples:
                raise ValueError(f"train_samples must be between 1 and {total_samples - 1}, got {train_samples}")

            train_val_samples = train_samples
            test_samples = total_samples - train_samples

            shuffle_indices = np.random.permutation(total_samples)
            train_val_indices = shuffle_indices[:train_val_samples]
            test_indices = shuffle_indices[train_val_samples:]
        else:
            test_samples = int(total_samples * test_size)
            shuffle_indices = np.random.permutation(total_samples)
            test_indices = shuffle_indices[:test_samples]
            train_val_indices = shuffle_indices[test_samples:]

        return {
            'train_val': {
                'features': features[train_val_indices],
                'logT': logT[train_val_indices],
                'N': N[train_val_indices]
            },
            'test': {
                'features': features[test_indices],
                'logT': logT[test_indices],
                'N': N[test_indices]
            },
            'raw_df': df
        }
    else:
        # 多组划分
        total_samples = len(features)
        splits = []

        for split_idx in range(n_splits):
            # 为每组划分使用不同的随机种子
            current_seed = random_seed + split_idx
            np.random.seed(current_seed)

            if train_samples is not None:
                if train_samples <= 0 or train_samples >= total_samples:
                    raise ValueError(f"train_samples must be between 1 and {total_samples - 1}, got {train_samples}")

                train_val_samples = train_samples
                test_samples = total_samples - train_samples

                shuffle_indices = np.random.permutation(total_samples)
                train_val_indices = shuffle_indices[:train_val_samples]
                test_indices = shuffle_indices[train_val_samples:]
            else:
                test_samples = int(total_samples * test_size)
                shuffle_indices = np.random.permutation(total_samples)
                test_indices = shuffle_indices[:test_samples]
                train_val_indices = shuffle_indices[test_samples:]

            splits.append({
                'split_idx': split_idx,
                'random_seed': current_seed,
                'train_val': {
                    'features': features[train_val_indices],
                    'logT': logT[train_val_indices],
                    'N': N[train_val_indices]
                },
                'test': {
                    'features': features[test_indices],
                    'logT': logT[test_indices],
                    'N': N[test_indices]
                },
                'raw_df': df.copy()  # 每个划分都需要自己的副本
            })

        return splits