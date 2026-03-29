from utils.utils_DNN_Structure import *
from utils.utils_Data import *
import random
import numpy as np
import os
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from pathlib import Path
torch.manual_seed(42)
np.random.seed(42)
"""
训练神经网络模块
====训练模块====
====外推模块====
"""

"""
================================================训练模块================================================
"""


def train_and_evaluate_with_hparams(
        file_path: str,
        best_hparams: dict,
        cluster_file_path: str = None,
        k_folds: int = 10,
        max_epochs: int = 200,
        patience: int = 20,
        output_dir: str = './results/02_NeuralNetwork',
        test_size: float = 0.2,

        seed_data_split: int = 42,
        seed_kfold: int = 123,
        seed_model_init: int = 456,
        seed_other: int = 789,

        random_seed: int = None
):
    # 🔧 兼容处理：如果传入了random_seed，则统一使用它（保持旧行为）
    if random_seed is not None:
        seed_data_split = seed_kfold = seed_model_init = seed_other = random_seed

    # ============================================== 辅助函数 ==============================================
    def set_random_seeds(seed: int):
        """统一设置所有随机种子，确保可复现性"""
        import torch
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ============================================== 数据加载 ==================================================
    # ✅ 使用 seed_data_split 控制数据划分
    data = load_data(file_path, test_size=test_size, random_seed=seed_data_split)

    if cluster_file_path is not None:
        cluster_data = load_data(cluster_file_path, test_size=test_size, random_seed=seed_data_split)
        # 数据统计
        original_train_len = len(data['train_val']['features'])
        original_test_len = len(data['test']['features'])
        cluster_train_len = len(cluster_data['train_val']['features'])
        cluster_test_len = len(cluster_data['test']['features'])
        # 合并数据
        data['train_val']['features'] = np.vstack([
            data['train_val']['features'],
            cluster_data['train_val']['features']
        ])
        data['train_val']['logT'] = np.concatenate([
            data['train_val']['logT'],
            cluster_data['train_val']['logT']
        ])
        data['train_val']['N'] = np.concatenate([
            data['train_val']['N'],
            cluster_data['train_val']['N']
        ])
        data['test']['features'] = np.vstack([
            data['test']['features'],
            cluster_data['test']['features']
        ])
        data['test']['logT'] = np.concatenate([
            data['test']['logT'],
            cluster_data['test']['logT']
        ])
        data['test']['N'] = np.concatenate([
            data['test']['N'],
            cluster_data['test']['N']
        ])

    # ============================================== 数据预处理 ==================================================
    all_features = np.vstack([data['train_val']['features'], data['test']['features']])
    all_logT = np.concatenate([data['train_val']['logT'], data['test']['logT']])
    f_mean = np.mean(all_features, axis=0)
    f_std = np.std(all_features, axis=0) + 1e-8
    logT_mean = np.mean(all_logT)
    logT_std = np.std(all_logT) + 1e-8

    data['train_val']['features_norm'] = (data['train_val']['features'] - f_mean) / f_std
    data['test']['features_norm'] = (data['test']['features'] - f_mean) / f_std
    data['train_val']['logT_norm'] = (data['train_val']['logT'] - logT_mean) / logT_std
    data['test']['logT_norm'] = (data['test']['logT'] - logT_mean) / logT_std

    features_norm = data['train_val']['features_norm']
    logT_norm = data['train_val']['logT_norm']
    logT = data['train_val']['logT']
    N_vals = data['train_val']['N']
    N_test_vals = data['test']['N']

    total = len(features_norm)
    # ✅ K折划分：使用独立种子 seed_kfold
    fold_indices = []
    indices = np.arange(total)
    np.random.seed(seed_kfold)
    np.random.shuffle(indices)

    fold_size = total // k_folds
    for fold in range(k_folds):
        if fold < k_folds - 1:
            val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        else:
            val_idx = indices[fold * fold_size:]
        train_idx = np.array([i for i in indices if i not in val_idx])
        fold_indices.append((train_idx, val_idx))

    per_fold_results = []
    fold_rms_summary = {}

    # ============================================== K折模型训练 ==================================================
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"Training fold ({fold + 1}/{k_folds})...")
        x_train = torch.FloatTensor(features_norm[train_idx])
        y_train = torch.FloatTensor(logT_norm[train_idx])
        x_val = torch.FloatTensor(features_norm[val_idx])
        y_val = torch.FloatTensor(logT_norm[val_idx])
        y_val_true = logT[val_idx]
        N_val = N_vals[val_idx]
        x_test = torch.FloatTensor(data['test']['features_norm'])
        y_test_true = data['test']['logT']
        N_test = N_test_vals

        # ✅ 模型初始化前：设置独立种子
        set_random_seeds(seed_model_init)
        model = AlphaDecayNN(5, int(best_hparams['hidden_layers']), int(best_hparams['hidden_neurons']))
        lambda_init = best_hparams.get('lambda_init', 1e-3)
        opt = AdaptiveLevenbergMarquardtOptimizer(model, lambda_=lambda_init, max_iter=max_epochs)

        best_val_rms = float('inf')
        no_improve = 0
        best_model_state = None

        for epoch in range(max_epochs):
            current_loss = opt.step(x_train, y_train)
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val)
                if val_outputs.dim() == 1:
                    val_outputs = val_outputs.unsqueeze(1)
                if y_val.dim() == 1:
                    y_val = y_val.unsqueeze(1)
                val_loss = torch.mean((val_outputs - y_val) ** 2).item()
                pred_log = (val_outputs.squeeze().numpy() * logT_std + logT_mean)
                val_rms = np.sqrt(np.mean((pred_log - y_val_true) ** 2))
            if val_rms < best_val_rms:
                best_val_rms = val_rms
                no_improve = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        with torch.no_grad():
            train_outputs = model(x_train)
            if train_outputs.dim() == 1:
                train_outputs = train_outputs.unsqueeze(1)
            y_tr_pred = (train_outputs.squeeze().numpy() * logT_std + logT_mean)

            val_outputs = model(x_val)
            if val_outputs.dim() == 1:
                val_outputs = val_outputs.unsqueeze(1)
            y_val_pred = (val_outputs.squeeze().numpy() * logT_std + logT_mean)

            test_outputs = model(x_test)
            if test_outputs.dim() == 1:
                test_outputs = test_outputs.unsqueeze(1)
            y_test_pred = (test_outputs.squeeze().numpy() * logT_std + logT_mean)

        tr_rms = np.sqrt(np.mean((y_tr_pred - logT[train_idx]) ** 2))
        val_rms = np.sqrt(np.mean((y_val_pred - y_val_true) ** 2))
        test_rms = np.sqrt(np.mean((y_test_pred - y_test_true) ** 2))

        fold_rms_summary[f'fold_{fold + 1}'] = {
            'train_rms': tr_rms,
            'val_rms': val_rms,
            'test_rms': test_rms
        }
        fold_result = {
            'train': {
                'N': N_vals[train_idx],
                'logT_exp': logT[train_idx],
                'logT_dl': y_tr_pred,
                'ratio': y_tr_pred - logT[train_idx]
            },
            'val': {
                'N': N_val,
                'logT_exp': y_val_true,
                'logT_dl': y_val_pred,
                'ratio': y_val_pred - y_val_true
            },
            'test': {
                'N': N_test,
                'logT_exp': y_test_true,
                'logT_dl': y_test_pred,
                'ratio': y_test_pred - y_test_true
            }
        }
        per_fold_results.append(fold_result)

    tr_rms_list = [fold_rms_summary[f]['train_rms'] for f in fold_rms_summary]
    val_rms_list = [fold_rms_summary[f]['val_rms'] for f in fold_rms_summary]
    test_rms_list = [fold_rms_summary[f]['test_rms'] for f in fold_rms_summary]
    print("FINAL LOG-SCALE RMS RESULTS")
    print(f"Training Set:   {np.mean(tr_rms_list):.4f} ± {np.std(tr_rms_list):.4f}")
    print(f"Validation Set: {np.mean(val_rms_list):.4f} ± {np.std(val_rms_list):.4f}")
    print(f"Test Set:       {np.mean(test_rms_list):.4f} ± {np.std(test_rms_list):.4f}")

    merged_N = np.concatenate([r['train']['N'] for r in per_fold_results] + [r['val']['N'] for r in per_fold_results])
    merged_exp = np.concatenate(
        [r['train']['logT_exp'] for r in per_fold_results] + [r['val']['logT_exp'] for r in per_fold_results])
    merged_dl = np.concatenate(
        [r['train']['logT_dl'] for r in per_fold_results] + [r['val']['logT_dl'] for r in per_fold_results])
    merged_ratio = np.concatenate(
        [r['train']['ratio'] for r in per_fold_results] + [r['val']['ratio'] for r in per_fold_results])
    merged_train_val_results = {
        'N': merged_N,
        'logT_exp': merged_exp,
        'logT_dl': merged_dl,
        'ratio': merged_ratio
    }
    test_results = per_fold_results[0]['test']

    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(merged_train_val_results).to_csv(
        os.path.join(output_dir, f'merged_train_val_predictions.csv'),
        index=False
    )
    pd.DataFrame(test_results).to_csv(
        os.path.join(output_dir, f'test_predictions.csv'),
        index=False
    )
    pd.DataFrame(fold_rms_summary).to_csv(
        os.path.join(output_dir, f'fold_rms_summary.csv'),
        index=False
    )

    # ============================================== 全量训练 ==================================================
    # ✅ 全量训练前也设置模型初始化种子（确保最终模型可复现）
    set_random_seeds(seed_model_init)
    final_model = AlphaDecayNN(5, int(best_hparams['hidden_layers']), int(best_hparams['hidden_neurons']))
    x_full = torch.FloatTensor(features_norm)
    y_full = torch.FloatTensor(logT_norm)
    lambda_init = best_hparams.get('lambda_init', 1e-3)
    opt = AdaptiveLevenbergMarquardtOptimizer(final_model, lambda_=lambda_init, max_iter=max_epochs)
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(max_epochs):
        current_loss = opt.step(x_full, y_full)
        if current_loss < best_loss:
            best_loss = current_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # ✅ 保存时记录所有种子配置
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'feature_mean': f_mean,
        'feature_std': f_std,
        'logT_mean': logT_mean,
        'logT_std': logT_std,
        'hidden_layers': best_hparams['hidden_layers'],
        'hidden_neurons': best_hparams['hidden_neurons'],
        'training_files': {
            'alpha': file_path,
            'cluster': cluster_file_path if cluster_file_path is not None else "None"
        },
        'hyperparameters': best_hparams,
        'data_params': {
            'test_size': test_size,
            'seed_data_split': seed_data_split,
            'seed_kfold': seed_kfold,
            'seed_model_init': seed_model_init,
            'seed_other': seed_other
        },
        'random_seeds': {  # ✅ 新增：显式记录所有种子
            'data_split': seed_data_split,
            'kfold': seed_kfold,
            'model_init': seed_model_init,
            'other': seed_other
        }
    }, os.path.join(output_dir, f'decay_model.pth'))

    print(f"\nModel saved to: {os.path.join(output_dir, f'decay_model.pth')}")

    return merged_train_val_results, test_results, fold_rms_summary, best_hparams


def train_and_evaluate_with_hparams_TL(
        file_path: str,
        best_hparams: dict,
        cluster_train_file: str = None,
        cluster_test_file: str = None,
        k_folds: int = 10,
        max_epochs: int = 200,
        patience: int = 20,
        output_dir: str = './results/02_NeuralNetwork',
        test_size: float = 0.2,

        # 🎲 独立随机种子参数（解耦关键！）
        seed_data_split: int = 42,  # 控制：Alpha数据的train/test划分
        seed_kfold: int = 123,  # 控制：K折交叉验证的折叠划分
        seed_model_init: int = 456,  # 控制：神经网络权重初始化
        seed_other: int = 789,  # 控制：其他随机操作（如dropout、aug等）

        # ⚠️ 兼容旧接口
        random_seed: int = None
):
    # 🔧 兼容处理
    if random_seed is not None:
        seed_data_split = seed_kfold = seed_model_init = seed_other = random_seed

    # ============================================== 辅助函数 ==============================================
    def set_random_seeds(seed: int):
        """统一设置所有随机种子，确保可复现性"""
        import torch
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def ensure_logT_exp(df, source_name):
        """如果缺失 logT_exp 列，通过 half_life_s 计算；否则直接使用"""
        if 'logT_exp' not in df.columns:
            if 'half_life_s' not in df.columns:
                raise ValueError(f"{source_name}: missing both 'logT_exp' and 'half_life_s' columns")
            df['logT_exp'] = np.log10(df['half_life_s'].astype(np.float64) + 1e-30)
            print(f"  → Computed logT_exp from half_life_s for {source_name}")
        else:
            print(f"  → Using existing logT_exp column for {source_name}")
        return df

    # ========== 模式判断与参数校验 ==========
    is_cluster_only = (cluster_train_file is None and cluster_test_file is not None)
    is_combined = (cluster_train_file is not None and cluster_test_file is not None)

    if not (is_cluster_only or is_combined):
        raise ValueError("Invalid configuration: must provide cluster_test_file. Pure alpha-only mode not supported.")

    training_mode = 'cluster_only' if is_cluster_only else 'combined'
    print(
        f"======================================== Training Mode: {training_mode.upper()} ========================================")
    print(f"Primary data file:   {file_path}")
    if is_combined:
        print(f"Cluster train file:  {cluster_train_file}")
    print(f"Cluster test file:   {cluster_test_file}\n")

    # ========== 数据加载（保留TL特有划分逻辑）==========
    print("======================================== Loading Data ========================================\n")

    if is_combined:
        # --- Combined 模式：加载 alpha 数据（80/20 划分）---
        df_alpha = pd.read_csv(file_path)
        df_alpha = ensure_logT_exp(df_alpha, "alpha data")

        features_alpha = df_alpha[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
        logT_alpha = df_alpha['logT_exp'].values.astype(np.float32)
        N_alpha = df_alpha['N'].values.astype(np.float32)

        # ✅ 使用 seed_data_split 控制Alpha数据划分
        np.random.seed(seed_data_split)
        indices = np.random.permutation(len(features_alpha))
        test_split = int(len(indices) * test_size)
        test_idx = indices[:test_split]
        train_idx = indices[test_split:]

        data = {
            'train_val': {
                'features': features_alpha[train_idx],
                'logT': logT_alpha[train_idx],
                'N': N_alpha[train_idx]
            },
            'test': {
                'features': features_alpha[test_idx],
                'logT': logT_alpha[test_idx],
                'N': N_alpha[test_idx]
            }
        }
        print(f"Loaded alpha data: {len(train_idx)} train/val + {len(test_idx)} test nuclei")

        # 合并 cluster 训练数据（100%进train_val）
        df_cluster_train = pd.read_csv(cluster_train_file)
        df_cluster_train = ensure_logT_exp(df_cluster_train, "cluster train data")

        features_cluster_train = df_cluster_train[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
        logT_cluster_train = df_cluster_train['logT_exp'].values.astype(np.float32)
        N_cluster_train = df_cluster_train['N'].values.astype(np.float32)

        data['train_val']['features'] = np.vstack([data['train_val']['features'], features_cluster_train])
        data['train_val']['logT'] = np.concatenate([data['train_val']['logT'], logT_cluster_train])
        data['train_val']['N'] = np.concatenate([data['train_val']['N'], N_cluster_train])
        print(f"Combined training set: {len(data['train_val']['features'])} nuclei")

    else:  # cluster_only 模式
        df_cluster_train = pd.read_csv(file_path)
        df_cluster_train = ensure_logT_exp(df_cluster_train, "cluster train data (file_path)")

        features_cluster_train = df_cluster_train[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
        logT_cluster_train = df_cluster_train['logT_exp'].values.astype(np.float32)
        N_cluster_train = df_cluster_train['N'].values.astype(np.float32)

        data = {
            'train_val': {
                'features': features_cluster_train,
                'logT': logT_cluster_train,
                'N': N_cluster_train
            },
            'test': {
                'features': np.empty((0, 5), dtype=np.float32),
                'logT': np.empty(0, dtype=np.float32),
                'N': np.empty(0, dtype=np.float32)
            }
        }
        print(f"Loaded cluster training data: {len(features_cluster_train)} nuclei (100% for K-fold)")

    # 加载 cluster 测试数据（100%进test）
    df_cluster_test = pd.read_csv(cluster_test_file)
    df_cluster_test = ensure_logT_exp(df_cluster_test, "cluster test data")

    features_cluster_test = df_cluster_test[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
    logT_cluster_test = df_cluster_test['logT_exp'].values.astype(np.float32)
    N_cluster_test = df_cluster_test['N'].values.astype(np.float32)

    # 合并到测试集
    if len(data['test']['features']) == 0:
        data['test']['features'] = features_cluster_test
        data['test']['logT'] = logT_cluster_test
        data['test']['N'] = N_cluster_test
    else:
        data['test']['features'] = np.vstack([data['test']['features'], features_cluster_test])
        data['test']['logT'] = np.concatenate([data['test']['logT'], logT_cluster_test])
        data['test']['N'] = np.concatenate([data['test']['N'], N_cluster_test])

    print(f"Loaded cluster testing data: {len(features_cluster_test)} nuclei (held-out for extrapolation)\n")

    # ========== 全局标准化（与参考函数完全一致）==========
    all_features = np.vstack([data['train_val']['features'], data['test']['features']])
    all_logT = np.concatenate([data['train_val']['logT'], data['test']['logT']])

    f_mean = np.mean(all_features, axis=0)
    f_std = np.std(all_features, axis=0) + 1e-8
    logT_mean = np.mean(all_logT)
    logT_std = np.std(all_logT) + 1e-8

    data['train_val']['features_norm'] = (data['train_val']['features'] - f_mean) / f_std
    data['test']['features_norm'] = (data['test']['features'] - f_mean) / f_std
    data['train_val']['logT_norm'] = (data['train_val']['logT'] - logT_mean) / logT_std
    data['test']['logT_norm'] = (data['test']['logT'] - logT_mean) / logT_std

    # ========== 准备K折划分变量（与参考函数完全一致）==========
    features_norm = data['train_val']['features_norm']
    logT_norm = data['train_val']['logT_norm']
    logT = data['train_val']['logT']
    N_vals = data['train_val']['N']
    N_test_vals = data['test']['N']

    total = len(features_norm)
    fold_indices = []
    indices = np.arange(total)

    # ✅ 使用 seed_kfold 控制K折划分
    np.random.seed(seed_kfold)
    np.random.shuffle(indices)

    fold_size = total // k_folds
    for fold in range(k_folds):
        if fold < k_folds - 1:
            val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        else:
            val_idx = indices[fold * fold_size:]
        train_idx = np.array([i for i in indices if i not in val_idx])
        fold_indices.append((train_idx, val_idx))

    per_fold_results = []
    fold_rms_summary = {}

    # ========== K折模型训练（与参考函数完全一致）==========
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"Training fold ({fold + 1}/{k_folds})...")

        x_train = torch.FloatTensor(features_norm[train_idx])
        y_train = torch.FloatTensor(logT_norm[train_idx])
        x_val = torch.FloatTensor(features_norm[val_idx])
        y_val = torch.FloatTensor(logT_norm[val_idx])
        y_val_true = logT[val_idx]
        N_val = N_vals[val_idx]
        x_test = torch.FloatTensor(data['test']['features_norm'])
        y_test_true = data['test']['logT']
        N_test = N_test_vals

        # ✅ 模型初始化前：设置独立种子
        set_random_seeds(seed_model_init)
        model = AlphaDecayNN(5, int(best_hparams['hidden_layers']), int(best_hparams['hidden_neurons']))
        lambda_init = best_hparams.get('lambda_init', 1e-3)
        opt = AdaptiveLevenbergMarquardtOptimizer(model, lambda_=lambda_init, max_iter=max_epochs)

        best_val_rms = float('inf')
        no_improve = 0
        best_model_state = None

        for epoch in range(max_epochs):
            current_loss = opt.step(x_train, y_train)
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val)
                if val_outputs.dim() == 1:
                    val_outputs = val_outputs.unsqueeze(1)
                if y_val.dim() == 1:
                    y_val = y_val.unsqueeze(1)
                val_loss = torch.mean((val_outputs - y_val) ** 2).item()
                pred_log = (val_outputs.squeeze().numpy() * logT_std + logT_mean)
                val_rms = np.sqrt(np.mean((pred_log - y_val_true) ** 2))
            if val_rms < best_val_rms:
                best_val_rms = val_rms
                no_improve = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        with torch.no_grad():
            train_outputs = model(x_train)
            if train_outputs.dim() == 1:
                train_outputs = train_outputs.unsqueeze(1)
            y_tr_pred = (train_outputs.squeeze().numpy() * logT_std + logT_mean)

            val_outputs = model(x_val)
            if val_outputs.dim() == 1:
                val_outputs = val_outputs.unsqueeze(1)
            y_val_pred = (val_outputs.squeeze().numpy() * logT_std + logT_mean)

            test_outputs = model(x_test)
            if test_outputs.dim() == 1:
                test_outputs = test_outputs.unsqueeze(1)
            y_test_pred = (test_outputs.squeeze().numpy() * logT_std + logT_mean)

        tr_rms = np.sqrt(np.mean((y_tr_pred - logT[train_idx]) ** 2))
        val_rms = np.sqrt(np.mean((y_val_pred - y_val_true) ** 2))
        test_rms = np.sqrt(np.mean((y_test_pred - y_test_true) ** 2))

        fold_rms_summary[f'fold_{fold + 1}'] = {
            'train_rms': tr_rms,
            'val_rms': val_rms,
            'test_rms': test_rms
        }

        fold_result = {
            'train': {
                'N': N_vals[train_idx],
                'logT_exp': logT[train_idx],
                'logT_dl': y_tr_pred,
                'ratio': y_tr_pred - logT[train_idx]
            },
            'val': {
                'N': N_val,
                'logT_exp': y_val_true,
                'logT_dl': y_val_pred,
                'ratio': y_val_pred - y_val_true
            },
            'test': {
                'N': N_test,
                'logT_exp': y_test_true,
                'logT_dl': y_test_pred,
                'ratio': y_test_pred - y_test_true
            }
        }
        per_fold_results.append(fold_result)

    # 【对齐】RMS结果汇总与打印
    tr_rms_list = [fold_rms_summary[f]['train_rms'] for f in fold_rms_summary]
    val_rms_list = [fold_rms_summary[f]['val_rms'] for f in fold_rms_summary]
    test_rms_list = [fold_rms_summary[f]['test_rms'] for f in fold_rms_summary]
    print("FINAL LOG-SCALE RMS RESULTS")
    print(f"Training Set:   {np.mean(tr_rms_list):.4f} ± {np.std(tr_rms_list):.4f}")
    print(f"Validation Set: {np.mean(val_rms_list):.4f} ± {np.std(val_rms_list):.4f}")
    print(f"Test Set:       {np.mean(test_rms_list):.4f} ± {np.std(test_rms_list):.4f}")

    # 【对齐】合并训练和验证集结果
    merged_N = np.concatenate([r['train']['N'] for r in per_fold_results] + [r['val']['N'] for r in per_fold_results])
    merged_exp = np.concatenate(
        [r['train']['logT_exp'] for r in per_fold_results] + [r['val']['logT_exp'] for r in per_fold_results])
    merged_dl = np.concatenate(
        [r['train']['logT_dl'] for r in per_fold_results] + [r['val']['logT_dl'] for r in per_fold_results])
    merged_ratio = np.concatenate(
        [r['train']['ratio'] for r in per_fold_results] + [r['val']['ratio'] for r in per_fold_results])
    merged_train_val_results = {
        'N': merged_N,
        'logT_exp': merged_exp,
        'logT_dl': merged_dl,
        'ratio': merged_ratio
    }
    test_results = per_fold_results[0]['test']

    # 【对齐】文件保存
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(merged_train_val_results).to_csv(
        os.path.join(output_dir, f'merged_train_val_predictions.csv'),
        index=False
    )
    pd.DataFrame(test_results).to_csv(
        os.path.join(output_dir, f'test_predictions.csv'),
        index=False
    )
    pd.DataFrame(fold_rms_summary).to_csv(
        os.path.join(output_dir, f'fold_rms_summary.csv'),
        index=False
    )

    # 【对齐】全量训练（带早停）
    print("\nFull training on entire train/val set...")

    # ✅ 全量训练前也设置模型初始化种子
    set_random_seeds(seed_model_init)
    final_model = AlphaDecayNN(5, int(best_hparams['hidden_layers']), int(best_hparams['hidden_neurons']))
    x_full = torch.FloatTensor(features_norm)
    y_full = torch.FloatTensor(logT_norm)
    lambda_init = best_hparams.get('lambda_init', 1e-3)
    opt = AdaptiveLevenbergMarquardtOptimizer(final_model, lambda_=lambda_init, max_iter=max_epochs)
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(max_epochs):
        current_loss = opt.step(x_full, y_full)
        if current_loss < best_loss:
            best_loss = current_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # 【对齐】模型保存内容（新增random_seeds字段）
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'feature_mean': f_mean,
        'feature_std': f_std,
        'logT_mean': logT_mean,
        'logT_std': logT_std,
        'hidden_layers': best_hparams['hidden_layers'],
        'hidden_neurons': best_hparams['hidden_neurons'],
        'training_files': {
            'alpha': file_path,
            'cluster': cluster_train_file if cluster_train_file else cluster_test_file
        },
        'hyperparameters': best_hparams,
        'data_params': {
            'test_size': test_size,
            'seed_data_split': seed_data_split,
            'seed_kfold': seed_kfold,
            'seed_model_init': seed_model_init,
            'seed_other': seed_other
        },
        'random_seeds': {  # ✅ 新增：显式记录所有种子
            'data_split': seed_data_split,
            'kfold': seed_kfold,
            'model_init': seed_model_init,
            'other': seed_other
        }
    }, os.path.join(output_dir, f'decay_model.pth'))

    print(f"\nModel saved to: {os.path.join(output_dir, f'decay_model.pth')}")

    return merged_train_val_results, test_results, fold_rms_summary, best_hparams


"""
================================================外推模块================================================
"""


def evaluate_single_subset_logscale_output_corrected(
        train_csv_path: str,
        test_csv_path: str,
        alpha_data_path: str,
        output_base_path: str,
        feature_cols: List[str] = ['Z', 'A', 'Q_MeV', 'Z_k', 'A_k'],
        hidden_layers: int = 1,
        hidden_neurons: int = 6,
        train_epochs: int = 200,
        random_seed: int = 42,
        batch_size: int = 32,
        optimizer_type: str = 'AdamW',  # 'AdamW' 或 'LM'
        lm_lambda: float = 1e-3  # LM优化器专用参数
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    专为绘图器优化的对数尺度输出：支持AdamW/LM双优化器
    【严格遵循您的要求】
      1. 使用全局定义的 AlphaDecayNN 网络
      2. 预测和反标准化严格使用 cluster 训练集的标准化参数

    输出CSV自动添加后缀:
      - ..._cluster_logscale_AdamW.csv
      - ..._combined_logscale_LM.csv
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # ==================== 元素符号映射 ====================
    ELEMENT_SYMBOLS = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
                       11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
                       19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
                       29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
                       37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh',
                       46: 'Pd',
                       47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
                       55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu',
                       64: 'Gd',
                       65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta',
                       74: 'W',
                       75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi',
                       84: 'Po',
                       85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np',
                       94: 'Pu',
                       95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr',
                       104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg',
                       112: 'Cn',
                       113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}

    # ==================== 数据加载 ====================
    alpha_df = pd.read_csv(alpha_data_path)
    if 'half_life_s_exp' in alpha_df.columns and 'half_life_s' not in alpha_df.columns:
        alpha_df['half_life_s'] = alpha_df['half_life_s_exp']
    alpha_df['Z_k'] = 2
    alpha_df['A_k'] = 4
    if alpha_df['Q_MeV'].dtype == object:
        def parse_q(q_val):
            q_str = str(q_val).strip()
            if '-' in q_str:
                parts = q_str.split('-')
                return (float(parts[0]) + float(parts[1])) / 2.0
            return float(q_str)

        alpha_df['Q_MeV'] = alpha_df['Q_MeV'].apply(parse_q)
    alpha_data_shared = alpha_df[feature_cols + ['half_life_s']].copy()
    alpha_data_shared['logT_exp'] = np.log10(alpha_data_shared['half_life_s'])

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    for df in [train_df, test_df]:
        if 'half_life_s_exp' in df.columns and 'half_life_s' not in df.columns:
            df['half_life_s'] = df['half_life_s_exp']
        if 'logT_exp' not in df.columns:
            df['logT_exp'] = np.log10(df['half_life_s'])
        if 'N' not in df.columns:
            df['N'] = df['A'] - df['Z']

    original_train_df = train_df.copy()
    original_test_df = test_df.copy()

    # ==================== 核心修正：使用全局 AlphaDecayNN ====================
    # 🔑 核心修正1: 删除内部SimpleNN，直接使用全局AlphaDecayNN
    # 注意：AlphaDecayNN 已在全局定义，此处直接调用

    # ==================== 核心修正：严格分离训练/预测标准化 ====================
    def train_and_predict_for_plotting(
            train_df_for_scaling: pd.DataFrame,  # ✅ 仅用于预测时的标准化参数（cluster训练集）
            actual_train_df: pd.DataFrame,  # 实际训练数据（决定模型学习的分布）
            predict_dfs: List[Tuple[pd.DataFrame, str]],
            feature_cols: List[str],
            epochs: int = 200,
            lr: float = 1e-3,
            random_seed: int = 42,
            optimizer_type: str = 'AdamW',
            lm_lambda: float = 1e-3
    ) -> pd.DataFrame:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # ========== 训练阶段：使用actual_train_df的标准化参数 ==========
        X_train_raw = actual_train_df[feature_cols].values.astype(np.float32)
        y_train_raw = actual_train_df['logT_exp'].values.astype(np.float32)

        # 计算训练数据的统计量（用于训练时标准化）
        feat_mean_train = np.mean(X_train_raw, axis=0)
        feat_std_train = np.std(X_train_raw, axis=0) + 1e-8
        logT_mean_train = np.mean(y_train_raw)
        logT_std_train = np.std(y_train_raw) + 1e-8

        # 标准化训练数据
        X_train_norm = (X_train_raw - feat_mean_train) / feat_std_train
        y_train_norm = (y_train_raw - logT_mean_train) / logT_std_train

        # 🔑 核心修正2: 使用全局AlphaDecayNN（替换SimpleNN）
        model = AlphaDecayNN(
            input_dim=len(feature_cols),  # 应为5
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons
        )

        # ============ 优化器选择 ============
        if optimizer_type == 'AdamW':
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            X_tensor = torch.FloatTensor(X_train_norm)
            y_tensor = torch.FloatTensor(y_train_norm)
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = criterion(model(X_tensor), y_tensor)
                loss.backward()
                optimizer.step()

        elif optimizer_type == 'LM':
            X_tensor = torch.FloatTensor(X_train_norm)
            y_tensor = torch.FloatTensor(y_train_norm).unsqueeze(1)
            lm_opt = AdaptiveLevenbergMarquardtOptimizer(
                model,
                lambda_=lm_lambda,
                max_iter=epochs
            )
            lm_opt.step(X_tensor, y_tensor)

        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Use 'AdamW' or 'LM'.")

        # ========== 预测阶段：严格使用cluster训练集的标准化参数 ==========
        model.eval()
        results = []
        with torch.no_grad():
            for target_df, dataset_type in predict_dfs:
                if len(target_df) == 0: continue

                # 🔑 核心修正3: 预测标准化参数严格来自cluster训练集（train_df_for_scaling）
                X_scaling = train_df_for_scaling[feature_cols].values.astype(np.float32)
                y_scaling = train_df_for_scaling['logT_exp'].values.astype(np.float32)

                feat_mean_pred = np.mean(X_scaling, axis=0)
                feat_std_pred = np.std(X_scaling, axis=0) + 1e-8
                logT_mean_pred = np.mean(y_scaling)
                logT_std_pred = np.std(y_scaling) + 1e-8

                # 用cluster参数标准化输入
                X_target_raw = target_df[feature_cols].values.astype(np.float32)
                X_target_norm = (X_target_raw - feat_mean_pred) / feat_std_pred  # ✅ 预测输入标准化

                # 模型预测
                y_pred_norm = model(torch.FloatTensor(X_target_norm)).numpy()

                # 🔑 核心修正4: 用cluster参数反标准化输出（严格遵循您的要求）
                y_pred_log = y_pred_norm * logT_std_pred + logT_mean_pred  # ✅ 预测输出反标准化

                # 计算差异
                y_true_log = target_df['logT_exp'].values.astype(np.float32)
                log_diffs = y_pred_log - y_true_log

                # 组装结果
                Z_vals = target_df['Z'].values
                A_vals = target_df['A'].values
                N_vals = target_df['N'].values
                for i in range(len(target_df)):
                    symbol = ELEMENT_SYMBOLS.get(int(Z_vals[i]), 'Xx')
                    nucleus_label = f"$^{{{int(A_vals[i])}}}${symbol}"

                    results.append({
                        'nucleus_label': nucleus_label,
                        'ratio': log_diffs[i],
                        'dataset_type': dataset_type,
                        'N': float(N_vals[i]),
                        'Z': float(Z_vals[i]),
                        'logT_exp': y_true_log[i],
                        'logT_pred': y_pred_log[i]
                    })
        return pd.DataFrame(results)

    # ==================== cluster_only 模式 ====================
    print(f"[plot-ready] Training cluster_only model with {optimizer_type} optimizer...")
    cluster_results = train_and_predict_for_plotting(
        train_df_for_scaling=original_train_df,  # ✅ 预测标准化参数源（cluster训练集）
        actual_train_df=original_train_df,  # ✅ 训练数据 = 标准化参数源（同源）
        predict_dfs=[(original_train_df, 'train'), (original_test_df, 'test')],
        feature_cols=feature_cols,
        epochs=train_epochs,
        lr=1e-3,
        random_seed=random_seed,
        optimizer_type=optimizer_type,
        lm_lambda=lm_lambda
    )
    cluster_results = cluster_results.sort_values(['N', 'Z']).reset_index(drop=True)
    co_overall_rms = np.sqrt(np.mean(cluster_results['ratio'] ** 2))
    print(f"[verify] cluster_only overall RMS ({optimizer_type}): {co_overall_rms:.6f}")

    # ==================== combined 模式 ====================
    print(f"[plot-ready] Training combined model with {optimizer_type} optimizer...")
    combined_train = pd.concat([
        original_train_df[feature_cols + ['logT_exp']],
        alpha_data_shared[feature_cols + ['logT_exp']]
    ], ignore_index=True)

    # 🔑 核心修正5: combined训练 + cluster预测标准化
    combined_results = train_and_predict_for_plotting(
        train_df_for_scaling=original_train_df,  # ✅ 预测标准化参数源（cluster训练集）
        actual_train_df=combined_train,  # ✅ 训练数据 = combined（源域+目标域）
        predict_dfs=[(original_train_df, 'train'), (original_test_df, 'test')],
        feature_cols=feature_cols,
        epochs=train_epochs,
        lr=1e-3,
        random_seed=random_seed,
        optimizer_type=optimizer_type,
        lm_lambda=lm_lambda
    )
    combined_results = combined_results.sort_values(['N', 'Z']).reset_index(drop=True)
    comb_overall_rms = np.sqrt(np.mean(combined_results['ratio'] ** 2))
    print(f"[verify] combined overall RMS ({optimizer_type}): {comb_overall_rms:.6f}")

    # ==================== 保存CSV（带优化器后缀） ====================
    output_dir = Path(output_base_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{optimizer_type}"
    cluster_csv = str(Path(output_base_path).with_suffix('')) + f"_cluster_logscale{suffix}.csv"
    combined_csv = str(Path(output_base_path).with_suffix('')) + f"_combined_logscale{suffix}.csv"

    required_cols = ['nucleus_label', 'ratio', 'dataset_type', 'N', 'Z', 'logT_exp', 'logT_pred']
    cluster_results[required_cols].to_csv(cluster_csv, index=False, encoding='utf-8-sig')
    combined_results[required_cols].to_csv(combined_csv, index=False, encoding='utf-8-sig')

    print(f"\n[output] Plot-ready CSVs saved with {optimizer_type} suffix:")
    print(f"  - Cluster-only: {cluster_csv}")
    print(f"  - Combined:     {combined_csv}")

    return cluster_results[required_cols], combined_results[required_cols]





def evaluate_single_subset_10fold_rms_output_corrected(
        train_csv_path: str,
        test_csv_path: str,
        alpha_data_path: str,
        output_base_path: str,
        feature_cols: List[str] = ['Z', 'A', 'Q_MeV', 'Z_k', 'A_k'],
        hidden_layers: int = 1,
        hidden_neurons: int = 6,
        train_epochs: int = 200,
        random_seed_base: int = 42,
        batch_size: int = 32,
        optimizer_type: str = 'AdamW',  # 'AdamW' 或 'LM'
        lm_lambda: float = 1e-3  # LM优化器专用参数
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    10折交叉验证RMS输出：支持AdamW/LM双优化器
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from pathlib import Path

    # ==================== 数据加载 ====================
    alpha_df = pd.read_csv(alpha_data_path)
    if 'half_life_s_exp' in alpha_df.columns and 'half_life_s' not in alpha_df.columns:
        alpha_df['half_life_s'] = alpha_df['half_life_s_exp']
    alpha_df['Z_k'] = 2
    alpha_df['A_k'] = 4
    if alpha_df['Q_MeV'].dtype == object:
        def parse_q(q_val):
            q_str = str(q_val).strip()
            if '-' in q_str:
                parts = q_str.split('-')
                return (float(parts[0]) + float(parts[1])) / 2.0
            return float(q_str)

        alpha_df['Q_MeV'] = alpha_df['Q_MeV'].apply(parse_q)
    alpha_data_shared = alpha_df[feature_cols + ['half_life_s']].copy()
    alpha_data_shared['logT_exp'] = np.log10(alpha_data_shared['half_life_s'])

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    for df in [train_df, test_df]:
        if 'half_life_s_exp' in df.columns and 'half_life_s' not in df.columns:
            df['half_life_s'] = df['half_life_s_exp']
        if 'logT_exp' not in df.columns:
            df['logT_exp'] = np.log10(df['half_life_s'])

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['logT_exp'].values.astype(np.float32)

    # ==================== 修正后的训练函数（支持双优化器） ====================
    def train_and_get_rms_corrected(
            X_train_cluster: np.ndarray,
            y_train_cluster: np.ndarray,
            X_train_combined: np.ndarray,
            y_train_combined: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            feature_cols: List[str],
            epochs: int = 200,
            lr: float = 1e-3,
            random_seed: int = 42,
            optimizer_type: str = 'AdamW',
            lm_lambda: float = 1e-3
    ) -> Tuple[float, float, float]:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # 标准化参数仅基于cluster训练集计算
        feat_scaler = StandardScaler()
        feat_scaler.fit(X_train_cluster)
        logT_scaler = StandardScaler()
        logT_scaler.fit(y_train_cluster.reshape(-1, 1))

        # 所有数据用同一套cluster标准化参数转换
        X_train_norm = feat_scaler.transform(X_train_combined)
        y_train_norm = logT_scaler.transform(y_train_combined.reshape(-1, 1)).flatten()

        # 构建模型
        class SimpleNN(nn.Module):
            def __init__(self, input_dim: int, hidden_layers: int, hidden_neurons: int):
                super().__init__()
                layers = [nn.Linear(input_dim, hidden_neurons), nn.ReLU()]
                for _ in range(hidden_layers - 1):
                    layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.ReLU()])
                layers.append(nn.Linear(hidden_neurons, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x): return self.net(x).squeeze()

        model = SimpleNN(len(feature_cols), hidden_layers, hidden_neurons)

        # ============ 优化器选择 ============
        if optimizer_type == 'AdamW':
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr)

            X_tensor = torch.FloatTensor(X_train_norm)
            y_tensor = torch.FloatTensor(y_train_norm)
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = criterion(model(X_tensor), y_tensor)
                loss.backward()
                optimizer.step()

        elif optimizer_type == 'LM':
            X_tensor = torch.FloatTensor(X_train_norm)
            y_tensor = torch.FloatTensor(y_train_norm).unsqueeze(1)

            lm_opt = AdaptiveLevenbergMarquardtOptimizer(
                model,
                lambda_=lm_lambda,
                max_iter=epochs
            )
            lm_opt.step(X_tensor, y_tensor)

        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Use 'AdamW' or 'LM'.")

        # 评估函数
        def compute_rms(X, y_true, feat_scaler, logT_scaler, model):
            if len(X) == 0:
                return np.nan
            X_norm = feat_scaler.transform(X)
            with torch.no_grad():
                y_pred_norm = model(torch.FloatTensor(X_norm)).numpy()
                y_pred = logT_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
            return np.sqrt(np.mean((y_pred - y_true) ** 2))

        model.eval()
        train_rms = compute_rms(X_train_cluster, y_train_cluster, feat_scaler, logT_scaler, model)
        val_rms = compute_rms(X_val, y_val, feat_scaler, logT_scaler, model) if len(X_val) > 0 else np.nan
        test_rms = compute_rms(X_test, y_test, feat_scaler, logT_scaler, model)

        return train_rms, val_rms, test_rms

    # ==================== 10折CV ====================
    print(f"[cv] Performing 10-fold CV with {optimizer_type} optimizer (CORRECTED standardization)")
    kf = KFold(n_splits=10, shuffle=True, random_state=random_seed_base)
    X_full_train = train_df[feature_cols].values.astype(np.float32)
    y_full_train = train_df['logT_exp'].values.astype(np.float32)

    cluster_rms = {'train': [], 'val': [], 'test': []}
    combined_rms = {'train': [], 'val': [], 'test': []}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full_train)):
        X_train_fold = X_full_train[train_idx]
        y_train_fold = y_full_train[train_idx]
        X_val_fold = X_full_train[val_idx]
        y_val_fold = y_full_train[val_idx]

        # cluster_only 模式
        co_train_rms, co_val_rms, co_test_rms = train_and_get_rms_corrected(
            X_train_cluster=X_train_fold,
            y_train_cluster=y_train_fold,
            X_train_combined=X_train_fold,
            y_train_combined=y_train_fold,
            X_val=X_val_fold,
            y_val=y_val_fold,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            epochs=train_epochs,
            lr=1e-3,
            random_seed=random_seed_base + fold_idx * 2,
            optimizer_type=optimizer_type,
            lm_lambda=lm_lambda
        )
        cluster_rms['train'].append(co_train_rms)
        cluster_rms['val'].append(co_val_rms)
        cluster_rms['test'].append(co_test_rms)

        # combined 模式
        X_train_combined = np.vstack([X_train_fold, alpha_data_shared[feature_cols].values.astype(np.float32)])
        y_train_combined = np.concatenate([y_train_fold, alpha_data_shared['logT_exp'].values.astype(np.float32)])

        comb_train_rms, comb_val_rms, comb_test_rms = train_and_get_rms_corrected(
            X_train_cluster=X_train_fold,
            y_train_cluster=y_train_fold,
            X_train_combined=X_train_combined,
            y_train_combined=y_train_combined,
            X_val=X_val_fold,
            y_val=y_val_fold,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            epochs=train_epochs,
            lr=1e-3,
            random_seed=random_seed_base + fold_idx * 2 + 1,
            optimizer_type=optimizer_type,
            lm_lambda=lm_lambda
        )
        combined_rms['train'].append(comb_train_rms)
        combined_rms['val'].append(comb_val_rms)
        combined_rms['test'].append(comb_test_rms)

        print(f"[fold {fold_idx + 1}/10] cluster_only test: {co_test_rms:.4f} | combined test: {comb_test_rms:.4f}")

    # ==================== 输出CSV（带优化器后缀） ====================
    fold_columns = [f"fold_{i + 1}" for i in range(10)]
    cluster_df = pd.DataFrame([cluster_rms['train'], cluster_rms['val'], cluster_rms['test']], columns=fold_columns)
    combined_df = pd.DataFrame([combined_rms['train'], combined_rms['val'], combined_rms['test']], columns=fold_columns)

    output_dir = Path(output_base_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{optimizer_type}"
    cluster_csv_path = str(Path(output_base_path).with_suffix('')) + f"_cluster{suffix}.csv"
    combined_csv_path = str(Path(output_base_path).with_suffix('')) + f"_combined{suffix}.csv"

    cluster_df.to_csv(cluster_csv_path, index=False, encoding='utf-8-sig')
    combined_df.to_csv(combined_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n[output] CORRECTED results saved with {optimizer_type} suffix:")
    print(f"  - Cluster-only: {cluster_csv_path}")
    print(f"  - Combined:     {combined_csv_path}")
    print(f"\n[summary] Mean test RMS (log₁₀ scale, {optimizer_type}):")
    print(f"  cluster_only: {np.mean(cluster_rms['test']):.4f} ± {np.std(cluster_rms['test']):.4f}")
    print(f"  combined:     {np.mean(combined_rms['test']):.4f} ± {np.std(combined_rms['test']):.4f}")

    return cluster_df, combined_df



"""
================================================种子评价模块================================================
"""


def evaluate_Seed_cluster(
        alpha_file_path: str,
        best_hparams: dict,
        train_csv_path: str,
        test_csv_path: str,
        output_dir: str,
        max_epochs: int = 200,
        patience: int = 20,
        random_seed: int = 42,
        k_folds: int = 10,
        test_size: float = 0.2,
        use_global: bool = True,  # True=K 折 + 全量，False=仅全量 ( recover 模式 )
):
    """评估函数：单种子 Cluster 训练与评估（Alpha 模式已移除）"""
    import torch, numpy as np, pandas as pd, os, random
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    # 随机种子设置工具函数
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)

    set_seed(random_seed)
    output_dir_path = Path(output_dir)
    os.makedirs(output_dir_path, exist_ok=True)

    def create_nucleus_label(Z: int, A: int) -> str:
        symbols = {56: 'Ba', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am',
                   96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf'}
        return f"$^{{{A}}}${symbols.get(Z, '?')}"

    def calc_rms(pred: np.ndarray, true: np.ndarray) -> float:
        diffs = pred - true
        valid = diffs[~np.isnan(diffs) & np.isfinite(diffs)]
        return float(np.sqrt(np.mean(valid ** 2))) if len(valid) > 0 else 0.0

    # ====================== Cluster 模式 ======================
    print(f"[Seed {random_seed}] Cluster training started.")
    train_df = pd.read_csv(train_csv_path).copy()
    test_df = pd.read_csv(test_csv_path).copy()
    required_cols = ['N', 'Z', 'A', 'Q_MeV', 'logT_exp', 'Emitted_Particle', 'Z_k', 'A_k']
    for col in required_cols:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in cluster data")
    train_df['dataset_type'], test_df['dataset_type'] = 'train', 'test'
    train_df['nucleus_label'] = train_df.apply(lambda r: create_nucleus_label(int(r['Z']), int(r['A'])), axis=1)
    test_df['nucleus_label'] = test_df.apply(lambda r: create_nucleus_label(int(r['Z']), int(r['A'])), axis=1)

    cluster_raw_df = pd.concat([train_df, test_df], ignore_index=True)
    cluster_features = cluster_raw_df[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
    cluster_logT_exp = cluster_raw_df['logT_exp'].values.astype(np.float32)
    cluster_train_mask = (cluster_raw_df['dataset_type'].values == 'train')

    def _train_and_evaluate_cluster(features, logT_exp, nuclei_info, train_mask, mode_name, seed):
        set_seed(seed)
        n_train, n_test = np.sum(train_mask), np.sum(~train_mask)
        if n_train < 5 or n_test < 2:
            return None, 0.0, 0.0, 0.0, None
        train_features, train_logT = features[train_mask], logT_exp[train_mask]
        if np.any(~np.isfinite(train_features)) or np.any(~np.isfinite(train_logT)):
            return None, 0.0, 0.0, 0.0, None
        f_mean_c, f_std_c = np.mean(train_features, 0), np.std(train_features, 0) + 1e-8
        logT_mean_c, logT_std_c = np.mean(train_logT), np.std(train_logT) + 1e-8
        f_std_c = np.where(f_std_c < 1e-10, 1.0, f_std_c)
        train_features_norm = (train_features - f_mean_c) / f_std_c
        train_logT_norm = (train_logT - logT_mean_c) / logT_std_c
        if np.any(~np.isfinite(train_features_norm)) or np.any(~np.isfinite(train_logT_norm)):
            return None, 0.0, 0.0, 0.0, None
        x_train_c, y_train_c = torch.FloatTensor(train_features_norm), torch.FloatTensor(train_logT_norm)
        model_c = AlphaDecayNN(input_dim=5, hidden_layers=int(best_hparams['hidden_layers']),
                               hidden_neurons=int(best_hparams['hidden_neurons']))
        opt_c = AdaptiveLevenbergMarquardtOptimizer(model_c, lambda_=best_hparams.get('lambda_init', 1e-3),
                                                    max_iter=max_epochs)
        best_val_loss, no_improve_c, best_model_state_c = float('inf'), 0, None
        use_val = len(x_train_c) > 20
        val_size_c = min(100, len(x_train_c) // 5) if use_val else 0
        x_val_c, y_val_c = (x_train_c[:val_size_c], y_train_c[:val_size_c]) if use_val else (None, None)
        for epoch in range(max_epochs):
            current_loss = opt_c.step(x_train_c, y_train_c)
            val_loss = None
            if use_val:
                model_c.eval()
                with torch.no_grad():
                    vp = model_c(x_val_c)
                    vp = vp.unsqueeze(1) if vp.dim() == 1 else vp
                    yt = y_val_c.unsqueeze(1) if y_val_c.dim() == 1 else y_val_c
                    val_loss = torch.mean((vp - yt) ** 2).item()
            target_loss = val_loss if use_val and val_loss is not None else current_loss
            if target_loss < best_val_loss:
                best_val_loss, no_improve_c = target_loss, 0
                best_model_state_c = {k: v.clone() for k, v in model_c.state_dict().items()}
            else:
                no_improve_c += 1
                if no_improve_c >= patience:
                    break
        if best_model_state_c:
            model_c.load_state_dict(best_model_state_c)
        model_c.eval()
        features_norm_c = (features - f_mean_c) / f_std_c
        with torch.no_grad():
            out = model_c(torch.FloatTensor(features_norm_c))
            out = out.unsqueeze(1) if out.dim() == 1 else out
            logT_pred = out.squeeze().numpy() * logT_std_c + logT_mean_c
        logT_pred = np.clip(np.nan_to_num(logT_pred, nan=0, posinf=300, neginf=-300), -300, 300)
        emitted = nuclei_info['Emitted_Particle'].values if 'Emitted_Particle' in nuclei_info.columns else np.array(
            [mode_name] * len(nuclei_info))
        results_df = pd.DataFrame({
            'N': nuclei_info['N'].values, 'Z': nuclei_info['Z'].values, 'A': nuclei_info['A'].values,
            'Q_MeV': nuclei_info['Q_MeV'].values, 'half_life_s_exp': 10 ** logT_exp, 'logT_exp': logT_exp,
            'logT_dl': logT_pred, 'ratio': logT_pred - logT_exp, 'dataset_type': nuclei_info['dataset_type'].values,
            'nucleus_label': nuclei_info['nucleus_label'].values, 'half_life_s_dl': 10 ** logT_pred,
            'Emitted_Particle': emitted
        })
        tm, ts = results_df['dataset_type'] == 'train', results_df['dataset_type'] == 'test'
        return results_df, calc_rms(results_df[tm]['logT_dl'], results_df[tm]['logT_exp']), \
            calc_rms(results_df[ts]['logT_dl'], results_df[ts]['logT_exp']), \
            calc_rms(results_df['logT_dl'], results_df['logT_exp']), model_c

    results_cluster, c_tr, c_te, c_tot, _ = _train_and_evaluate_cluster(
        cluster_features, cluster_logT_exp, cluster_raw_df, cluster_train_mask, 'cluster', random_seed)
    results_cluster.to_csv(output_dir_path / 'results_cluster.csv', index=False)
    print(f"cluster - Train RMS: {c_tr:.4f}, Test RMS: {c_te:.4f}, Total RMS: {c_tot:.4f}")

    # 保存指标（Alpha 指标留空，由 evaluate_Seed_alpha 补充）
    pd.DataFrame([
        {'seed': random_seed, 'mode': 'alpha', 'train_rms': np.nan, 'test_rms': np.nan, 'total_rms': np.nan,
         'use_global': use_global},
        {'seed': random_seed, 'mode': 'cluster', 'train_rms': c_tr, 'test_rms': c_te, 'total_rms': c_tot,
         'use_global': use_global}
    ]).to_csv(output_dir_path / 'metrics_summary.csv', index=False)
    print(f"[Seed {random_seed}] Cluster training completed.")

    return {
        'cluster': {'results': results_cluster, 'train_rms': c_tr, 'test_rms': c_te, 'total_rms': c_tot}
    }

def evaluate_Seed_alpha(
        alpha_file_path: str,
        best_hparams: dict,
        output_dir: str,
        cluster_file_path: str = None,
        k_folds: int = 10,
        max_epochs: int = 200,
        patience: int = 20,
        test_size: float = 0.2,
        seed: int = 42  # 🔑 控制模型初始化等随机性，但不影响数据划分
):
    """
    训练与评估函数：支持通过 seed 参数研究随机初始化对模型的影响
    保证：
    1. 数据划分固定使用 random_state=42，确保不同 seed 实验的数据集内容一致
    2. 全量训练使用 train_val（训练集 + 验证集），不包含测试集
    3. seed 仅影响：模型权重初始化、K 折索引、优化器随机性
    4. 新增：生成 results_alpha.csv 文件
    """
    import torch, numpy as np, pandas as pd, os, random
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    # ============================================== 随机种子设置 ==============================================
    def set_seed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(s)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)

    set_seed(seed)
    output_dir_path = Path(output_dir)
    os.makedirs(output_dir_path, exist_ok=True)

    def create_nucleus_label(Z: int, A: int) -> str:
        symbols = {56: 'Ba', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am',
                   96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf'}
        return f"$^{{{A}}}${symbols.get(Z, '?')}"

    def calc_rms(pred: np.ndarray, true: np.ndarray) -> float:
        diffs = pred - true
        valid = diffs[~np.isnan(diffs) & np.isfinite(diffs)]
        return float(np.sqrt(np.mean(valid ** 2))) if len(valid) > 0 else 0.0

    # ============================================== 数据加载 ==================================================
    # 🔑 关键：数据划分固定使用 random_state=42，不受 seed 参数影响
    alpha_df = pd.read_csv(alpha_file_path)
    for col in ['N', 'Z', 'A', 'Q_MeV', 'Z_k', 'A_k', 'half_life_s']:
        if col not in alpha_df.columns:
            raise ValueError(f"Missing column '{col}' in alpha data")
    if 'logT_exp' not in alpha_df.columns:
        alpha_df['logT_exp'] = np.log10(alpha_df['half_life_s'])
    if 'nucleus_label' not in alpha_df.columns:
        alpha_df['nucleus_label'] = alpha_df.apply(lambda r: create_nucleus_label(int(r['Z']), int(r['A'])), axis=1)
    if 'Emitted_Particle' not in alpha_df.columns:
        alpha_df['Emitted_Particle'] = 'alpha'

    # 数据划分（固定种子 42）
    features_raw = alpha_df[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
    logT_raw = alpha_df['logT_exp'].values.astype(np.float32)
    features_train_val, features_test, logT_train_val, logT_test, idx_train_val, idx_test = train_test_split(
        features_raw, logT_raw, np.arange(len(alpha_df)),
        test_size=test_size, random_state=42
    )

    if cluster_file_path is not None:
        cluster_df = pd.read_csv(cluster_file_path)
        cluster_features = cluster_df[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
        cluster_logT = cluster_df['logT_exp'].values.astype(np.float32)
        features_train_val = np.vstack([features_train_val, cluster_features])
        logT_train_val = np.concatenate([logT_train_val, cluster_logT])

    # ============================================== 数据预处理（全局标准化）==================================
    all_features = np.vstack([features_train_val, features_test])
    all_logT = np.concatenate([logT_train_val, logT_test])
    f_mean = np.mean(all_features, axis=0)
    f_std = np.std(all_features, axis=0) + 1e-8
    logT_mean = np.mean(all_logT)
    logT_std = np.std(all_logT) + 1e-8

    features_train_val_norm = (features_train_val - f_mean) / f_std
    features_test_norm = (features_test - f_mean) / f_std
    logT_train_val_norm = (logT_train_val - logT_mean) / logT_std
    logT_test_norm = (logT_test - logT_mean) / logT_std

    # ============================================== K 折交叉验证 ==============================================
    total = len(features_train_val_norm)
    fold_indices = []
    indices = np.arange(total)
    np.random.seed(seed)
    np.random.shuffle(indices)

    fold_size = total // k_folds
    for fold in range(k_folds):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size] if fold < k_folds - 1 else indices[fold * fold_size:]
        train_idx = np.array([i for i in indices if i not in val_idx])
        fold_indices.append((train_idx, val_idx))

    per_fold_results, fold_rms_summary = [], {}

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"Training fold ({fold + 1}/{k_folds})...")
        x_train = torch.FloatTensor(features_train_val_norm[train_idx])
        y_train = torch.FloatTensor(logT_train_val_norm[train_idx])
        x_val = torch.FloatTensor(features_train_val_norm[val_idx])
        y_val_true = logT_train_val[val_idx]
        x_test = torch.FloatTensor(features_test_norm)
        y_test_true = logT_test

        model = AlphaDecayNN(5, int(best_hparams['hidden_layers']), int(best_hparams['hidden_neurons']))
        opt = AdaptiveLevenbergMarquardtOptimizer(model, lambda_=best_hparams.get('lambda_init', 1e-3),
                                                  max_iter=max_epochs)

        best_val_rms, no_improve, best_model_state = float('inf'), 0, None
        for epoch in range(max_epochs):
            current_loss = opt.step(x_train, y_train)
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                if val_pred.dim() == 1:
                    val_pred = val_pred.unsqueeze(1)
                pred_log = val_pred.squeeze().numpy() * logT_std + logT_mean
                val_rms = np.sqrt(np.mean((pred_log - y_val_true) ** 2))
            if val_rms < best_val_rms:
                best_val_rms, no_improve = val_rms, 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    if best_model_state:
                        model.load_state_dict(best_model_state)
                    break
        if best_model_state:
            model.load_state_dict(best_model_state)

        model.eval()
        with torch.no_grad():
            def _predict(x):
                out = model(x)
                if out.dim() == 1:
                    out = out.unsqueeze(1)
                return out.squeeze().numpy() * logT_std + logT_mean

            y_tr_pred, y_val_pred, y_te_pred = _predict(x_train), _predict(x_val), _predict(x_test)

        fold_rms_summary[f'fold_{fold + 1}'] = {
            'train_rms': np.sqrt(np.mean((y_tr_pred - logT_train_val[train_idx]) ** 2)),
            'val_rms': np.sqrt(np.mean((y_val_pred - y_val_true) ** 2)),
            'test_rms': np.sqrt(np.mean((y_te_pred - y_test_true) ** 2))
        }

    # K 折结果汇总
    for label, lst in [('Training', [fold_rms_summary[f]['train_rms'] for f in fold_rms_summary]),
                       ('Validation', [fold_rms_summary[f]['val_rms'] for f in fold_rms_summary]),
                       ('Test', [fold_rms_summary[f]['test_rms'] for f in fold_rms_summary])]:
        print(f"{label} Set: {np.mean(lst):.4f} ± {np.std(lst):.4f}")

    # 保存 K 折结果
    pd.DataFrame(fold_rms_summary).to_csv(output_dir_path / 'fold_rms_summary.csv', index=False)

    # ============================================== 全量训练（train_val 数据）=================================
    print(f"[Seed {seed}] Full-data training with train_val data...")

    x_full = torch.FloatTensor(features_train_val_norm)
    y_full = torch.FloatTensor(logT_train_val_norm)

    final_model = AlphaDecayNN(5, int(best_hparams['hidden_layers']), int(best_hparams['hidden_neurons']))
    opt = AdaptiveLevenbergMarquardtOptimizer(final_model, lambda_=best_hparams.get('lambda_init', 1e-3),
                                              max_iter=max_epochs)

    best_loss, no_improve = float('inf'), 0
    for epoch in range(max_epochs):
        current_loss = opt.step(x_full, y_full)
        if current_loss < best_loss:
            best_loss, no_improve = current_loss, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # 保存模型
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'feature_mean': f_mean, 'feature_std': f_std,
        'logT_mean': logT_mean, 'logT_std': logT_std,
        'hidden_layers': best_hparams['hidden_layers'],
        'hidden_neurons': best_hparams['hidden_neurons'],
        'training_files': {'alpha': alpha_file_path, 'cluster': cluster_file_path if cluster_file_path else "None"},
        'hyperparameters': best_hparams,
        'data_params': {'test_size': test_size, 'seed': seed, 'split_seed': 42}
    }, output_dir_path / f'decay_model_seed{seed}.pth')
    print(f"Model saved to: {output_dir_path / f'decay_model_seed{seed}.pth'}")

    # ============================================== 生成 results_alpha.csv ====================================
    final_model.eval()
    with torch.no_grad():
        def _predict_full(x):
            out = final_model(x)
            if out.dim() == 1:
                out = out.unsqueeze(1)
            return out.squeeze().numpy() * logT_std + logT_mean

        y_train_pred_full = _predict_full(x_full)
        y_test_pred_full = _predict_full(torch.FloatTensor(features_test_norm))

    a_tr = calc_rms(y_train_pred_full, logT_train_val)
    a_te = calc_rms(y_test_pred_full, logT_test)
    a_tot = calc_rms(np.concatenate([y_train_pred_full, y_test_pred_full]), np.concatenate([logT_train_val, logT_test]))

    # 构建 Alpha 结果 DataFrame
    all_idx = np.concatenate([idx_train_val, idx_test])
    results_alpha = pd.DataFrame({
        'N': alpha_df.iloc[all_idx]['N'].values,
        'Z': alpha_df.iloc[all_idx]['Z'].values,
        'A': alpha_df.iloc[all_idx]['A'].values,
        'Q_MeV': alpha_df.iloc[all_idx]['Q_MeV'].values,
        'half_life_s_exp': 10 ** np.concatenate([logT_train_val, logT_test]),
        'logT_exp': np.concatenate([logT_train_val, logT_test]),
        'logT_dl': np.concatenate([y_train_pred_full, y_test_pred_full]),
        'ratio': np.concatenate([y_train_pred_full, y_test_pred_full]) - np.concatenate([logT_train_val, logT_test]),
        'dataset_type': ['train'] * len(logT_train_val) + ['test'] * len(logT_test),
        'nucleus_label': alpha_df.iloc[all_idx]['nucleus_label'].values,
        'half_life_s_dl': 10 ** np.concatenate([y_train_pred_full, y_test_pred_full]),
        'Emitted_Particle': alpha_df.iloc[all_idx]['Emitted_Particle'].values
    })
    results_alpha.to_csv(output_dir_path / 'results_alpha.csv', index=False)
    print(f"alpha - Train RMS: {a_tr:.4f}, Test RMS: {a_te:.4f}, Total RMS: {a_tot:.4f}")

    # ============================================== 更新 metrics_summary.csv ==================================
    # 读取 evaluate_Seed_cluster 生成的 metrics_summary.csv，补充 Alpha 指标
    metrics_path = output_dir_path / 'metrics_summary.csv'
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        # 更新 Alpha 行
        alpha_idx = metrics_df[metrics_df['mode'] == 'alpha'].index
        if len(alpha_idx) > 0:
            metrics_df.loc[alpha_idx[0], 'train_rms'] = a_tr
            metrics_df.loc[alpha_idx[0], 'test_rms'] = a_te
            metrics_df.loc[alpha_idx[0], 'total_rms'] = a_tot
            metrics_df.to_csv(metrics_path, index=False)
            print(f"[Seed {seed}] metrics_summary.csv updated with Alpha metrics.")

    return {
        'fold_rms_summary': fold_rms_summary,
        'full_data_rms': a_tot,
        'alpha_train_rms': a_tr,
        'alpha_test_rms': a_te,
        'alpha_total_rms': a_tot,
        'model_path': str(output_dir_path / f'model_alpha_seed{seed}.pth'),
        'best_hparams': best_hparams
    }