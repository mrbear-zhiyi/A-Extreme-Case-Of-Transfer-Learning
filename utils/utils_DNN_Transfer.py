"""
===============================================================================
DNN TRANSFER LEARNING MODULE - Nuclear Decay Prediction with Transfer Learning
===============================================================================

This module provides comprehensive transfer learning capabilities for nuclear
decay half-life prediction, supporting both cluster decay and alpha decay modes.

Key Features:
- Transfer learning from pre-trained alpha decay models
- Subset training with k-fold validation (k=3~10)
- Full training mode for complete dataset
- Model ensemble support for uncertainty estimation
- Automatic dimension matching for pretrained models
- Layer freezing for fine-tuning control

Dependencies:
- torch>=1.9.0
- numpy>=1.21.0
- pandas>=1.4.0
- scikit-learn>=0.24.0

Author: [Your Name]
Date: 2026
"""
from torch import combinations

from utils.utils_DNN_Structure import *
from typing import Callable, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pathlib import Path
import config
import random
import time
import os

torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# TRANSFER LEARNING FUNCTIONS
# =============================================================================

def transfer_learn_from_preselected_indices_all(
        alpha_model_path: str,
        cluster_data: pd.DataFrame,
        Cluster: dict,
        output_dir: str,
        freeze_first: int = None,
        max_epochs: int = 100,
        patience: int = 20,
        use_ensemble: bool = False,
        ensemble_seeds: list = [42, 123, 456],
        lambda_init: float = 1e-3,
        overall_rms_summary_csv_path: str = None,
        full_training: bool = False
):
    """
    Evaluate transfer learning model performance with subset training (k=3~10) and full training.

    Parameters
    ----------
    alpha_model_path : str
        Path to pre-trained alpha decay model
    cluster_data : pd.DataFrame
        Cluster decay dataset with required columns
    Cluster : dict
        Dictionary containing preselected cluster indices for each k value
    output_dir : str
        Output directory for saving results
    freeze_first : int, optional
        Number of hidden layers to freeze during training
    max_epochs : int, optional
        Maximum number of training epochs (default: 100)
    patience : int, optional
        Early stopping patience (default: 20)
    use_ensemble : bool, optional
        Whether to use ensemble training (default: False)
    ensemble_seeds : list, optional
        Random seeds for ensemble training (default: [42, 123, 456])
    lambda_init : float, optional
        Initial lambda for LM optimizer (default: 1e-3)
    overall_rms_summary_csv_path : str, optional
        Path to save RMS summary CSV
    full_training : bool, optional
        Whether to perform full training on complete dataset (default: False)

    Returns
    -------
    best_experiment_results : dict
        Results from the best performing experiment
    rms_df : pd.DataFrame
        DataFrame containing RMS records for all experiments
    """
    base_output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    required_columns = ['Z', 'N', 'A', 'Q_MeV', 'half_life_s', 'Emitted_Particle', 'Z_k', 'A_k']
    missing_cols = [col for col in required_columns if col not in cluster_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    features = cluster_data[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
    logT = np.log10(cluster_data['half_life_s'].values.astype(np.float32))
    N_vals = cluster_data['N'].values.astype(np.float32)
    n_total = len(cluster_data)

    f_mean, f_std = features.mean(0), features.std(0) + 1e-8
    logT_mean, logT_std = logT.mean(), logT.std() + 1e-8
    features_norm = (features - f_mean) / f_std
    logT_norm = (logT - logT_mean) / logT_std

    ckpt = torch.load(alpha_model_path, map_location='cpu', weights_only=False)
    state_dict = load_pretrained_model_with_dimension_match(ckpt['model_state_dict'], current_input_dim=5)
    hidden_layers = ckpt.get('hidden_layers', 2)
    hidden_neurons = ckpt.get('hidden_neurons', 64)

    def train_single(seed, x_train, y_train, x_all, test_idx=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = AlphaDecayNN(5, hidden_layers, hidden_neurons)
        model.load_state_dict(state_dict)
        if freeze_first:
            freeze_model_layers(model, num_frozen_hidden_layers=freeze_first)
        opt = AdaptiveLevenbergMarquardtOptimizer(model, lambda_=lambda_init, max_iter=max_epochs)
        best_val_rms = float('inf')
        no_improve = 0
        best_model_state = None

        for epoch in range(max_epochs):
            opt.step(x_train, y_train)
            model.eval()
            with torch.no_grad():
                if test_idx is not None and len(test_idx) > 0:
                    val_pred = model(x_all[test_idx]).squeeze().numpy() * logT_std + logT_mean
                    val_rms = np.sqrt(np.mean((val_pred - logT[test_idx]) ** 2))
                else:
                    train_pred = model(x_train).squeeze().numpy() * logT_std + logT_mean
                    val_rms = np.sqrt(np.mean((train_pred - (y_train.numpy() * logT_std + logT_mean)) ** 2)) * 1.05

            if val_rms < best_val_rms:
                best_val_rms = val_rms
                no_improve = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif no_improve >= patience:
                break
            else:
                no_improve += 1

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            full_pred = model(x_all).squeeze().numpy() * logT_std + logT_mean
        return full_pred, model, best_val_rms

    rms_records = []
    best_overall_rms = float('inf')
    best_experiment_results = None
    x_all = torch.FloatTensor(features_norm)

    max_combinations = getattr(config, 'CLUSTER_INDICES', 200)
    print(f"Training with top {max_combinations} subsets per k-value (from config.CLUSTER_INDICES)")

    print(f"\n{'=' * 60}")
    print("SUBSET TRAINING (k=3~10) - Executing for all models")
    print(f"{'=' * 60}")

    for k in [3, 4, 5, 6, 7, 8, 9, 10]:
        if k not in Cluster:
            print(f"  Skipping k={k}: no preselected subsets available")
            continue
        valid_count = 0
        for row_id, res in enumerate(Cluster[k][:max_combinations]):
            train_idx = res['Cluster_Indices']
            if not isinstance(train_idx, list) or not train_idx or any(not (0 <= i < n_total) for i in train_idx):
                continue
            test_idx = np.setdiff1d(np.arange(n_total), train_idx).tolist()
            x_train = torch.FloatTensor(features_norm[train_idx])
            y_train = torch.FloatTensor(logT_norm[train_idx])

            if use_ensemble:
                preds = []
                models = []
                rms_values = []
                for seed in ensemble_seeds:
                    pred, model, test_rms_val = train_single(seed, x_train, y_train, x_all, test_idx)
                    preds.append(pred)
                    models.append(model)
                    rms_values.append(test_rms_val)
                final_pred = np.mean(preds, axis=0)
                best_model_idx = np.argmin(rms_values)
                final_model = models[best_model_idx]
                best_test_rms = rms_values[best_model_idx]
            else:
                final_pred, final_model, best_test_rms = train_single(42, x_train, y_train, x_all, test_idx)

            train_rms = np.sqrt(np.mean((final_pred[train_idx] - logT[train_idx]) ** 2))
            test_rms = np.sqrt(np.mean((final_pred[test_idx] - logT[test_idx]) ** 2)) if test_idx else np.nan
            overall_rms = np.sqrt(np.mean((final_pred - logT) ** 2))

            rms_records.append({
                'k': k, 'row_id': row_id, 'train_size': len(train_idx),
                'train_sigma_rms': train_rms, 'test_sigma_rms': test_rms, 'overall_sigma_rms': overall_rms,
                'train_indices': str(train_idx), 'test_indices': str(test_idx)
            })

            train_results = {
                'N': N_vals[train_idx],
                'Z': cluster_data.iloc[train_idx]['Z'].values,
                'A': cluster_data.iloc[train_idx]['A'].values,
                'Q_MeV': cluster_data.iloc[train_idx]['Q_MeV'].values,
                'logT_exp': logT[train_idx],
                'logT_dl': final_pred[train_idx],
                'ratio': final_pred[train_idx] - logT[train_idx],
                'half_life_s_exp': cluster_data.iloc[train_idx]['half_life_s'].values,
                'Emitted_Particle': cluster_data.iloc[train_idx]['Emitted_Particle'].values,
                'Z_k': cluster_data.iloc[train_idx]['Z_k'].values,
                'A_k': cluster_data.iloc[train_idx]['A_k'].values
            }
            test_results = {
                'N': N_vals[test_idx],
                'Z': cluster_data.iloc[test_idx]['Z'].values,
                'A': cluster_data.iloc[test_idx]['A'].values,
                'Q_MeV': cluster_data.iloc[test_idx]['Q_MeV'].values,
                'logT_exp': logT[test_idx],
                'logT_dl': final_pred[test_idx],
                'ratio': final_pred[test_idx] - logT[test_idx],
                'half_life_s_exp': cluster_data.iloc[test_idx]['half_life_s'].values,
                'Emitted_Particle': cluster_data.iloc[test_idx]['Emitted_Particle'].values,
                'Z_k': cluster_data.iloc[test_idx]['Z_k'].values,
                'A_k': cluster_data.iloc[test_idx]['A_k'].values
            }

            pd.DataFrame(train_results).to_csv(
                os.path.join(output_dir, f"train_k{k}_row{row_id}.csv"),
                index=False
            )
            pd.DataFrame(test_results).to_csv(
                os.path.join(output_dir, f"test_k{k}_row{row_id}.csv"),
                index=False
            )

            torch.save({
                'model_state_dict': final_model.state_dict(),
                'feature_mean': f_mean, 'feature_std': f_std,
                'logT_mean': logT_mean, 'logT_std': logT_std,
                'train_indices': train_idx, 'test_indices': test_idx,
                'k': k, 'row_id': row_id, 'lambda_init': lambda_init,
                'patience': patience, 'use_ensemble': use_ensemble,
                'hidden_layers': hidden_layers, 'hidden_neurons': hidden_neurons,
                'full_training': False
            }, os.path.join(output_dir, f"transfer_k{k}_row{row_id}.pth"))

            if overall_rms < best_overall_rms:
                best_overall_rms = overall_rms
                best_experiment_results = {
                    'train_results': train_results,
                    'test_results': test_results,
                    'final_pred': final_pred,
                    'train_idx': train_idx,
                    'test_idx': test_idx
                }
            valid_count += 1
            if (row_id + 1) % 50 == 0:
                print(f"  k={k}: Completed {row_id + 1}/{min(max_combinations, len(Cluster[k]))} subsets")
        print(f"  k={k}: Completed {valid_count} valid subsets")

    if full_training:
        print(f"\n{'=' * 60}")
        print(f"FULL TRAINING MODE ACTIVATED (k={n_total})")
        print(f"{'=' * 60}")
        full_output_dir = os.path.join(output_dir, "full_training")
        os.makedirs(full_output_dir, exist_ok=True)
        print(f"Full training results will be saved to: {full_output_dir}")

        train_idx = list(range(n_total))
        test_idx = []
        x_train = torch.FloatTensor(features_norm[train_idx])
        y_train = torch.FloatTensor(logT_norm[train_idx])

        if use_ensemble:
            preds = []
            models = []
            rms_values = []
            for seed in ensemble_seeds:
                pred, model, val_rms = train_single(seed, x_train, y_train, x_all, test_idx)
                preds.append(pred)
                models.append(model)
                rms_values.append(val_rms)
            final_pred = np.mean(preds, axis=0)
            best_model_idx = np.argmin(rms_values)
            final_model = models[best_model_idx]
            best_val_rms = rms_values[best_model_idx]
        else:
            final_pred, final_model, best_val_rms = train_single(42, x_train, y_train, x_all, test_idx)

        train_rms = np.sqrt(np.mean((final_pred[train_idx] - logT[train_idx]) ** 2))
        test_rms = np.nan
        overall_rms = train_rms

        rms_records.append({
            'k': n_total,
            'row_id': 0,
            'train_size': len(train_idx),
            'train_sigma_rms': train_rms,
            'test_sigma_rms': test_rms,
            'overall_sigma_rms': overall_rms,
            'train_indices': str(train_idx),
            'test_indices': str(test_idx)
        })

        train_results = {
            'N': N_vals[train_idx],
            'Z': cluster_data.iloc[train_idx]['Z'].values,
            'A': cluster_data.iloc[train_idx]['A'].values,
            'Q_MeV': cluster_data.iloc[train_idx]['Q_MeV'].values,
            'logT_exp': logT[train_idx],
            'logT_dl': final_pred[train_idx],
            'ratio': final_pred[train_idx] - logT[train_idx],
            'half_life_s_exp': cluster_data.iloc[train_idx]['half_life_s'].values,
            'Emitted_Particle': cluster_data.iloc[train_idx]['Emitted_Particle'].values,
            'Z_k': cluster_data.iloc[train_idx]['Z_k'].values,
            'A_k': cluster_data.iloc[train_idx]['A_k'].values
        }
        test_results = {
            'N': np.array([]),
            'Z': np.array([]),
            'A': np.array([]),
            'Q_MeV': np.array([]),
            'logT_exp': np.array([]),
            'logT_dl': np.array([]),
            'ratio': np.array([]),
            'half_life_s_exp': np.array([]),
            'Emitted_Particle': np.array([]),
            'Z_k': np.array([]),
            'A_k': np.array([])
        }

        pd.DataFrame(train_results).to_csv(
            os.path.join(full_output_dir, "train_full.csv"),
            index=False
        )
        pd.DataFrame(test_results).to_csv(
            os.path.join(full_output_dir, "test_full.csv"),
            index=False
        )

        torch.save({
            'model_state_dict': final_model.state_dict(),
            'feature_mean': f_mean, 'feature_std': f_std,
            'logT_mean': logT_mean, 'logT_std': logT_std,
            'train_indices': train_idx, 'test_indices': test_idx,
            'k': n_total, 'row_id': 0,
            'lambda_init': lambda_init, 'patience': patience,
            'use_ensemble': use_ensemble,
            'hidden_layers': hidden_layers, 'hidden_neurons': hidden_neurons,
            'full_training': True
        }, os.path.join(full_output_dir, "transfer_full.pth"))

        print(f"Full training completed | Train RMS: {train_rms:.4f} | Overall RMS: {overall_rms:.4f}")
        print(f"Results saved to: {full_output_dir}")

    rms_df = pd.DataFrame()
    if rms_records:
        rms_df = pd.DataFrame(rms_records)
        rms_df.to_csv(os.path.join(output_dir, 'transfer_rms_summary.csv'), index=False)

    if overall_rms_summary_csv_path:
        subset_rms = rms_df[rms_df['k'].isin([3, 4, 5, 6, 7, 8, 9, 10])]
        if not subset_rms.empty:
            k_avg = subset_rms.groupby('k')['overall_sigma_rms'].nsmallest(3).groupby('k').mean()
            k_avg_df = k_avg.reset_index(name='avg_rms')
            k_avg_df.to_csv(overall_rms_summary_csv_path, index=False)

    if best_experiment_results is not None:
        if len(best_experiment_results['test_results']['N']) > 0:
            all_results = {
                'N': np.concatenate([best_experiment_results['train_results']['N'],
                                     best_experiment_results['test_results']['N']]),
                'Z': np.concatenate([best_experiment_results['train_results']['Z'],
                                     best_experiment_results['test_results']['Z']]),
                'A': np.concatenate([best_experiment_results['train_results']['A'],
                                     best_experiment_results['test_results']['A']]),
                'Q_MeV': np.concatenate([best_experiment_results['train_results']['Q_MeV'],
                                         best_experiment_results['test_results']['Q_MeV']]),
                'logT_exp': np.concatenate([best_experiment_results['train_results']['logT_exp'],
                                            best_experiment_results['test_results']['logT_exp']]),
                'logT_dl': np.concatenate([best_experiment_results['train_results']['logT_dl'],
                                           best_experiment_results['test_results']['logT_dl']]),
                'ratio': np.concatenate([best_experiment_results['train_results']['ratio'],
                                         best_experiment_results['test_results']['ratio']]),
                'half_life_s_exp': np.concatenate([best_experiment_results['train_results']['half_life_s_exp'],
                                                   best_experiment_results['test_results']['half_life_s_exp']]),
                'Emitted_Particle': np.concatenate([best_experiment_results['train_results']['Emitted_Particle'],
                                                    best_experiment_results['test_results']['Emitted_Particle']]),
                'Z_k': np.concatenate([best_experiment_results['train_results']['Z_k'],
                                       best_experiment_results['test_results']['Z_k']]),
                'A_k': np.concatenate([best_experiment_results['train_results']['A_k'],
                                       best_experiment_results['test_results']['A_k']])
            }
        else:
            all_results = best_experiment_results['train_results']
        pd.DataFrame(all_results).to_csv(
            os.path.join(output_dir, 'all_predictions.csv'),
            index=False
        )

    print(f"\n{'=' * 60}")
    print(f"TRANSFER LEARNING COMPLETED")
    print(f"{'=' * 60}")
    print(f"✓ Subset training (k=3~10): Completed with top {max_combinations} subsets per k")
    if full_training:
        print(f"✓ Full training (k={n_total}): Completed in 'full_training' subdirectory")
    print(f"Best subset RMS: {best_overall_rms:.4f} (from k=3~10 experiments)")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")

    return best_experiment_results, rms_df


def transfer_learn_from_alpha_model(
        alpha_model_path: str,
        train_csv_path: str,
        test_csv_path: str,
        output_dir: str,
        freeze_first: int = None,
        max_epochs: int = 200,
        patience: int = 20,
        lambda_init: float = 1e-3,
        use_early_stopping: bool = True,
):
    """
    Transfer learning from Alpha model with fixed 5-feature input and global standardization.

    Parameters
    ----------
    alpha_model_path : str
        Path to pre-trained alpha decay model
    train_csv_path : str
        Path to training dataset CSV
    test_csv_path : str
        Path to test dataset CSV
    output_dir : str
        Output directory for saving results
    freeze_first : int, optional
        Number of hidden layers to freeze
    max_epochs : int, optional
        Maximum training epochs (default: 200)
    patience : int, optional
        Early stopping patience (default: 20)
    lambda_init : float, optional
        Initial lambda for LM optimizer (default: 1e-3)
    use_early_stopping : bool, optional
        Whether to enable early stopping (default: True)

    Returns
    -------
    results_dict : dict
        Dictionary containing train, test, and all results
    rms_dict : dict
        Dictionary containing RMS metrics
    """
    TRANSFER_SEED = 42
    random.seed(TRANSFER_SEED)
    np.random.seed(TRANSFER_SEED)
    torch.manual_seed(TRANSFER_SEED)
    torch.cuda.manual_seed(TRANSFER_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(TRANSFER_SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = pd.read_csv(train_csv_path), pd.read_csv(test_csv_path)
    required_columns = ['N', 'Z', 'A', 'Q_MeV', 'Z_k', 'A_k', 'half_life_s_exp', 'Emitted_Particle']
    for df_name, df in [('train', train_df), ('test', test_df)]:
        missing = [c for c in required_columns if c not in df.columns]
        if missing: raise ValueError(f"{df_name} CSV missing required columns: {missing}")
    if 'dataset_type' not in train_df.columns: train_df['dataset_type'] = 'train'
    if 'dataset_type' not in test_df.columns: test_df['dataset_type'] = 'test'
    cluster_data = pd.concat([train_df, test_df], ignore_index=True)

    FEATURE_COLS = ['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']
    features = cluster_data[FEATURE_COLS].values.astype(np.float32)
    logT = np.log10(cluster_data['half_life_s_exp'].values.astype(np.float32))
    N_vals = cluster_data['N'].values.astype(np.float32)
    train_idx = np.where(cluster_data['dataset_type'] == 'train')[0].tolist()
    test_idx = np.where(cluster_data['dataset_type'] == 'test')[0].tolist()
    print(f"  Dataset: {len(train_idx)} train, {len(test_idx)} test | Features: {FEATURE_COLS}")

    f_mean, f_std = features.mean(0), features.std(0) + 1e-8
    logT_mean, logT_std = logT.mean(), logT.std() + 1e-8
    features_norm = (features - f_mean) / f_std
    logT_norm = (logT - logT_mean) / logT_std

    print(f"  Loading Alpha model: {alpha_model_path}")
    ckpt = torch.load(alpha_model_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        hidden_layers = ckpt.get('hidden_layers', 2)
        hidden_neurons = ckpt.get('hidden_neurons', 64)
    else:
        state_dict = ckpt if isinstance(ckpt, dict) else None
        if state_dict is None: raise ValueError("Unrecognized checkpoint format")
        hidden_layers, hidden_neurons = 2, 64
        print(f"  ⚠️  Warning: Loading pure state_dict. Using default architecture.")

    INPUT_DIM = 5
    print(f"  Model: input_dim={INPUT_DIM}, hidden_layers={hidden_layers}, hidden_neurons={hidden_neurons}")

    def train_model(x_train, y_train, x_all):
        random.seed(TRANSFER_SEED)
        np.random.seed(TRANSFER_SEED)
        torch.manual_seed(TRANSFER_SEED)
        torch.cuda.manual_seed(TRANSFER_SEED)
        model = AlphaDecayNN(INPUT_DIM, hidden_layers, hidden_neurons)
        model.load_state_dict(state_dict)
        if freeze_first: freeze_model_layers(model, num_frozen_hidden_layers=freeze_first)
        opt = AdaptiveLevenbergMarquardtOptimizer(model, lambda_=lambda_init, max_iter=max_epochs)
        best_val_rms, no_improve, best_state = float('inf'), 0, None
        actual_epochs_run = 0

        for epoch in range(max_epochs):
            actual_epochs_run += 1
            opt.step(x_train, y_train)
            model.eval()
            with torch.no_grad():
                idx = test_idx if len(test_idx) > 0 else train_idx
                pred = model(x_all[idx]).squeeze().numpy() * logT_std + logT_mean
                true = logT[idx]
                val_rms = np.sqrt(np.mean((pred - true) ** 2))

            if val_rms < best_val_rms:
                best_val_rms, no_improve = val_rms, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif use_early_stopping and no_improve >= patience:
                print(f"  Early Stopping triggered at epoch {actual_epochs_run}")
                break
            else:
                no_improve += 1

        print(f"  Training finished after {actual_epochs_run} epochs (Max: {max_epochs})")

        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            full_pred = model(x_all).squeeze().numpy() * logT_std + logT_mean
        return full_pred, model, best_val_rms

    x_all = torch.FloatTensor(features_norm)
    x_train = torch.FloatTensor(features_norm[train_idx])
    y_train = torch.FloatTensor(logT_norm[train_idx])
    final_pred, final_model, best_val_rms = train_model(x_train, y_train, x_all)

    train_rms = np.sqrt(np.mean((final_pred[train_idx] - logT[train_idx]) ** 2))
    test_rms = np.sqrt(np.mean((final_pred[test_idx] - logT[test_idx]) ** 2)) if test_idx else np.nan
    overall_rms = np.sqrt(np.mean((final_pred - logT) ** 2))
    print(f"  Train RMS: {train_rms:.4f} | Test RMS: {test_rms:.4f} | Overall RMS: {overall_rms:.4f}")

    def _build_results(idx, dtype):
        res = {
            'N': N_vals[idx], 'Z': cluster_data.iloc[idx]['Z'].values, 'A': cluster_data.iloc[idx]['A'].values,
            'Q_MeV': cluster_data.iloc[idx]['Q_MeV'].values, 'Z_k': cluster_data.iloc[idx]['Z_k'].values,
            'A_k': cluster_data.iloc[idx]['A_k'].values, 'logT_exp': logT[idx], 'logT_dl': final_pred[idx],
            'ratio': final_pred[idx] - logT[idx], 'half_life_s_exp': cluster_data.iloc[idx]['half_life_s_exp'].values,
            'Emitted_Particle': cluster_data.iloc[idx]['Emitted_Particle'].values, 'dataset_type': [dtype] * len(idx)
        }
        return res

    train_res, test_res = _build_results(train_idx, 'train'), _build_results(test_idx, 'test')
    pd.DataFrame(train_res).to_csv(os.path.join(output_dir, "train_results.csv"), index=False)
    pd.DataFrame(test_res).to_csv(os.path.join(output_dir, "test_results.csv"), index=False)
    all_res = {k: np.concatenate([train_res[k], test_res[k]]) for k in train_res if k in test_res}
    pd.DataFrame(all_res).to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)

    torch.save({
        'model_state_dict': final_model.state_dict(),
        'feature_mean': f_mean, 'feature_std': f_std, 'logT_mean': logT_mean, 'logT_std': logT_std,
        'train_indices': train_idx, 'test_indices': test_idx, 'lambda_init': lambda_init, 'patience': patience,
        'hidden_layers': hidden_layers, 'hidden_neurons': hidden_neurons, 'input_dim': INPUT_DIM,
        'feature_names': FEATURE_COLS, 'source_alpha_model': alpha_model_path, 'transfer_seed': TRANSFER_SEED,
        'normalization_type': 'global',
        'use_early_stopping': use_early_stopping
    }, os.path.join(output_dir, "transfer_model.pth"))

    rms_dict = {'train_rms': train_rms, 'test_rms': test_rms, 'overall_rms': overall_rms, 'val_rms': best_val_rms,
                'train_size': len(train_idx), 'test_size': len(test_idx), 'transfer_seed': TRANSFER_SEED,
                'normalization_type': 'global', 'use_early_stopping': use_early_stopping}
    pd.DataFrame([rms_dict]).to_csv(os.path.join(output_dir, "rms_metrics.csv"), index=False)

    return {'train_results': train_res, 'test_results': test_res, 'all_results': all_res,
            'model': final_model}, rms_dict


def select_best_by_mad_similarity(
        all_results: list,
        top_k: int = 10,
        ratio_col: str = 'ratio',
        cluster_size: int = None
) -> list:
    """
    Select most compact cluster of seeds based on pairwise Euclidean distance.

    Parameters
    ----------
    all_results : list
        List of all seed results
    top_k : int, optional
        Number of seeds to select (default: 10)
    ratio_col : str, optional
        Column name for ratio values (default: 'ratio')
    cluster_size : int, optional
        Size of compact cluster (default: None, uses top_k)

    Returns
    -------
    selected_seeds : list
        List of selected seed IDs forming the most compact cluster
    """
    if cluster_size is None:
        cluster_size = top_k
        print(f"[Path B] cluster_size not specified, using top_k={top_k}")
    else:
        print(f"[Path B] Finding most compact cluster of {cluster_size} seeds...")

    seed_ratio_vectors = {}
    common_keys = None

    for item in all_results:
        seed = item['seed']
        df = pd.DataFrame(item['results_dict']['all_results'])
        df_sorted = df.sort_values(['N', 'Z', 'Emitted_Particle']).reset_index(drop=True)
        keys = list(zip(df_sorted['N'], df_sorted['Z'], df_sorted['Emitted_Particle']))
        ratios = df_sorted[ratio_col].values

        if common_keys is None:
            common_keys = keys
            seed_ratio_vectors[seed] = ratios
        else:
            mask = [k in common_keys for k in keys]
            seed_ratio_vectors[seed] = ratios[mask]

    seeds = list(seed_ratio_vectors.keys())
    if len(seeds) < cluster_size:
        print(f"[Warning] Only {len(seeds)} seeds available, less than cluster_size={cluster_size}")
        return seeds

    feature_matrix = np.array([seed_ratio_vectors[s] for s in seeds])
    n_seeds = len(seeds)
    print(f"[Path B] Feature matrix: {feature_matrix.shape}")

    print(f"[Path B] Computing pairwise Euclidean distances...")
    norms = np.sum(feature_matrix ** 2, axis=1)
    dist_matrix = norms[:, np.newaxis] + norms[np.newaxis, :] - 2 * np.dot(feature_matrix, feature_matrix.T)
    dist_matrix = np.sqrt(np.maximum(dist_matrix, 0))

    best_group = None
    best_avg_internal_dist = float('inf')

    for center_idx in range(n_seeds):
        distances = dist_matrix[center_idx].copy()
        distances[center_idx] = np.inf
        nearest_indices = np.argsort(distances)[:cluster_size - 1]
        group_indices = [center_idx] + list(nearest_indices)

        internal_dists = []
        for i, j in combinations(group_indices, 2):
            internal_dists.append(dist_matrix[i, j])

        avg_internal_dist = np.mean(internal_dists)

        if avg_internal_dist < best_avg_internal_dist:
            best_avg_internal_dist = avg_internal_dist
            best_group = [seeds[idx] for idx in group_indices]

    log_data = []
    for seed in seeds:
        if seed in best_group:
            idx = seeds.index(seed)
            group_indices = [seeds.index(s) for s in best_group]
            avg_to_others = np.mean([dist_matrix[idx, gi] for gi in group_indices if gi != idx])
            log_data.append({
                'seed': seed,
                'avg_dist_to_group': avg_to_others,
                'selected': True,
                'cluster_size': cluster_size
            })
        else:
            log_data.append({
                'seed': seed,
                'avg_dist_to_group': None,
                'selected': False,
                'cluster_size': cluster_size
            })

    pd.DataFrame(log_data).to_csv(
        Path(all_results[0]['output_dir']).parent / '.compact_cluster_log.csv',
        index=False
    )

    print(f"\n[Path B] Most compact cluster found ({cluster_size} seeds, avg internal distance = {best_avg_internal_dist:.4f}):")
    for i, s in enumerate(best_group, 1):
        idx = seeds.index(s)
        avg_to_group = np.mean([dist_matrix[idx, seeds.index(gs)] for gs in best_group if gs != s])
        print(f"  {i}. Seed {s:5d} | avg dist to group = {avg_to_group:.4f}")

    return best_group


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def generate_cluster_prediction_summary(
        model_dir: str,
        full_cluster_csv: str,
        output_csv: str,
        deduplicate: bool = True
) -> pd.DataFrame:
    """
    Generate prediction summary for cluster decay using the best model.

    Parameters
    ----------
    model_dir : str
        Directory containing trained models
    full_cluster_csv : str
        Path to full cluster decay dataset
    output_csv : str
        Output CSV path for prediction summary
    deduplicate : bool, optional
        Whether to remove duplicate entries (default: True)

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame containing prediction results and statistics
    """
    rms_summary_path = os.path.join(model_dir, 'transfer_rms_summary.csv')
    if not os.path.exists(rms_summary_path):
        raise FileNotFoundError(f"RMS summary file not found: {rms_summary_path}")

    rms_df = pd.read_csv(rms_summary_path)
    full_model_path = os.path.join(model_dir, 'transfer_full.pth')

    if os.path.exists(full_model_path):
        best_model_path = full_model_path
        print("Full training model detected, will use full training model for prediction")
    else:
        if rms_df.empty:
            raise ValueError("RMS summary is empty, cannot determine best model")
        best_row = rms_df.loc[rms_df['overall_sigma_rms'].idxmin()]
        k_val = int(best_row['k'])
        row_id = int(best_row['row_id'])
        best_model_path = os.path.join(model_dir, f'transfer_k{k_val}_row{row_id}.pth')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model file not found: {best_model_path}")
        print(f"Selected best model: k={k_val}, row={row_id} | Overall RMS: {best_row['overall_sigma_rms']:.4f}")

    print(f"Loading model: {os.path.basename(best_model_path)}")
    ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
    hidden_layers = ckpt.get('hidden_layers', 2)
    hidden_neurons = ckpt.get('hidden_neurons', 64)
    model = AlphaDecayNN(5, hidden_layers, hidden_neurons)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    f_mean = ckpt['feature_mean']
    f_std = ckpt['feature_std']
    logT_mean = ckpt['logT_mean']
    logT_std = ckpt['logT_std']

    print(f"Loading full cluster data: {full_cluster_csv}")
    cluster_df = pd.read_csv(full_cluster_csv)
    required_cols = ['Z', 'N', 'A', 'Q_MeV', 'half_life_s', 'Emitted_Particle', 'Z_k', 'A_k']
    missing = [col for col in required_cols if col not in cluster_df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    if cluster_df['Q_MeV'].dtype == object:
        def parse_q_value(q_val):
            q_str = str(q_val).strip()
            if '-' in q_str:
                parts = q_str.split('-')
                try:
                    return (float(parts[0]) + float(parts[1])) / 2.0
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid Q_MeV range format: '{q_val}'")
            else:
                return float(q_str)

        original_q_count = len(cluster_df[cluster_df['Q_MeV'].astype(str).str.contains('-', na=False)])
        cluster_df['Q_MeV'] = cluster_df['Q_MeV'].apply(parse_q_value)
        if original_q_count > 0:
            print(f"Processed {original_q_count} Q_MeV range values (using mid-point)")

    if 'certainty' not in cluster_df.columns:
        print("Warning: 'certainty' column missing, defaulting to True for all entries")
        cluster_df['certainty'] = True
    else:
        cluster_df['certainty'] = cluster_df['certainty'].astype(str).str.strip().str.upper() == 'TRUE'
        true_count = cluster_df['certainty'].sum()
        false_count = len(cluster_df) - true_count
        print(f"Certainty distribution: {true_count} certain, {false_count} uncertain entries")

    certain_mask = cluster_df['certainty']
    uncertain_mask = ~cluster_df['certainty']
    if certain_mask.any() and uncertain_mask.any():
        certain_min = cluster_df.loc[certain_mask, 'half_life_s'].min()
        uncertain_max = cluster_df.loc[uncertain_mask, 'half_life_s'].max()
        if certain_min > 1e10 and uncertain_max < 1000:
            raise ValueError(
                "Inconsistent half_life_s format detected:\n"
                f"  - Certain data min: {certain_min:.2e} seconds (valid)\n"
                f"  - Uncertain data max: {uncertain_max:.2f} (likely log10 values)\n"
                "For uncertain data with log10 lower bounds (e.g., >25.9),\n"
                "please convert to seconds using 10^value before prediction.\n"
                "Example: log10(T/s) > 25.9  =>  half_life_s = 10^25.9"
            )

    dedup_cols = ['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']
    original_count = len(cluster_df)
    if deduplicate:
        cluster_df = cluster_df.sort_values('certainty', ascending=False)
        cluster_df = cluster_df.drop_duplicates(subset=dedup_cols, keep='first').reset_index(drop=True)
        print(f"Deduplication: {original_count} -> {len(cluster_df)} records (based on {dedup_cols})")
    else:
        print(f"Deduplication disabled, keeping all {original_count} records")

    features = cluster_df[['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']].values.astype(np.float32)
    logT_exp = np.log10(cluster_df['half_life_s'].values.astype(np.float32))
    N_vals = cluster_df['N'].values.astype(np.float32)
    features_norm = (features - f_mean) / f_std

    with torch.no_grad():
        x_tensor = torch.FloatTensor(features_norm)
        logT_pred_norm = model(x_tensor).squeeze().numpy()
        logT_pred = logT_pred_norm * logT_std + logT_mean

    safe_logT_pred = np.clip(logT_pred, -300, 300)
    half_life_pred = 10 ** safe_logT_pred

    element_symbols = {
        87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
        93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf',
        99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf'
    }
    nucleus_labels = []
    for z, a in zip(cluster_df['Z'], cluster_df['A']):
        symbol = element_symbols.get(int(z), '?')
        nucleus_labels.append(f"$^{{{int(a)}}}${symbol}")

    results_df = pd.DataFrame({
        'nucleus_label': nucleus_labels,
        'Z': cluster_df['Z'].values,
        'A': cluster_df['A'].values,
        'N': cluster_df['N'].values,
        'Q_MeV': cluster_df['Q_MeV'].values,
        'Emitted_Particle': cluster_df['Emitted_Particle'].values,
        'Z_k': cluster_df['Z_k'].values,
        'A_k': cluster_df['A_k'].values,
        'half_life_s_exp': cluster_df['half_life_s'].values,
        'logT_exp': logT_exp,
        'half_life_s_pred': half_life_pred,
        'logT_pred': logT_pred,
        'difference_logT': logT_pred - logT_exp,
        'certainty': cluster_df['certainty'].values
    })

    results_df['half_life_s_exp_display'] = results_df.apply(
        lambda row: f">{row['half_life_s_exp']:.2e}" if not row['certainty'] else f"{row['half_life_s_exp']:.2e}",
        axis=1
    )

    results_df = results_df[[
        'nucleus_label', 'Z', 'A', 'N', 'Q_MeV', 'Emitted_Particle', 'Z_k', 'A_k',
        'half_life_s_exp_display', 'logT_exp', 'half_life_s_pred', 'logT_pred',
        'difference_logT', 'certainty'
    ]]

    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nPrediction summary completed!")
    print(f"  - Input records: {original_count} -> Deduplicated: {len(results_df)}")
    print(f"  - Output file: {output_csv}")
    print(f"  - Statistics:")
    print(f"    * Mean absolute difference_logT: {np.mean(np.abs(results_df['difference_logT'])):.4f}")
    print(f"    * certainty=true samples: {(results_df['certainty'] == True).sum()}")
    print(f"    * certainty=false samples: {(results_df['certainty'] == False).sum()}")

    return results_df


def evaluate_subsets_independent_training(
        rms_summary_path: str,
        alpha_data_path: str,
        model_1_6_dir: str,
        output_csv_path: str,
        feature_cols: List[str] = ['Z', 'A', 'Q_MeV', 'Z_k', 'A_k'],
        hidden_layers: int = 1,
        hidden_neurons: int = 6,
        train_epochs: int = 200,
        random_seed_base: int = 42,
        batch_size: int = 32
) -> pd.DataFrame:
    """
    Evaluate 400 subsets with independent training (800 models total).

    Parameters
    ----------
    rms_summary_path : str
        Path to RMS summary CSV
    alpha_data_path : str
        Path to alpha decay dataset
    model_1_6_dir : str
        Directory containing train/test CSV files
    output_csv_path : str
        Output path for results CSV
    feature_cols : list, optional
        Feature column names (default: ['Z', 'A', 'Q_MeV', 'Z_k', 'A_k'])
    hidden_layers : int, optional
        Number of hidden layers (default: 1)
    hidden_neurons : int, optional
        Number of hidden neurons (default: 6)
    train_epochs : int, optional
        Training epochs (default: 200)
    random_seed_base : int, optional
        Base random seed (default: 42)
    batch_size : int, optional
        Training batch size (default: 32)

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame containing evaluation results for all subsets
    """
    print("[processing] Loading alpha decay data...")
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

    print("[processing] Selecting subsets...")
    rms_df = pd.read_csv(rms_summary_path)
    if 'k' not in rms_df.columns and 'clusters' in rms_df.columns:
        rms_df['k'] = rms_df['clusters']
    if 'overall_sigma_rms' not in rms_df.columns and 'rms' in rms_df.columns:
        rms_df['overall_sigma_rms'] = rms_df['rms']

    top_subsets = []
    for k_val in range(3, 11):
        k_df = rms_df[rms_df['k'] == k_val].copy()
        n_select = min(100, len(k_df))
        best_50 = k_df.nsmallest(n_select, 'overall_sigma_rms')
        for _, row in best_50.iterrows():
            top_subsets.append({
                'k': int(row['k']),
                'row_id': int(row['row_id']),
                'rms': float(row['overall_sigma_rms'])
            })

    class SimpleNN(nn.Module):
        def __init__(self, input_dim: int, hidden_layers: int, hidden_neurons: int):
            super().__init__()
            layers = []
            layers.append(nn.Linear(input_dim, hidden_neurons))
            layers.append(nn.ReLU())
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_neurons, hidden_neurons))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_neurons, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze()

    def train_and_evaluate_single_model(
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            feature_cols: List[str],
            model_name: str,
            epochs: int = 200,
            lr: float = 1e-3,
            random_seed: int = 42,
            batch_size: int = 32
    ) -> Tuple[float, float, float]:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = np.log10(np.maximum(train_df['half_life_s'].values.astype(np.float32), 1e-10))
        X_test = test_df[feature_cols].values.astype(np.float32) if len(test_df) > 0 else np.array([])
        y_test = np.log10(np.maximum(test_df['half_life_s'].values.astype(np.float32), 1e-10)) if len(
            test_df) > 0 else np.array([])

        feat_scaler = StandardScaler()
        X_train_norm = feat_scaler.fit_transform(X_train)
        logT_scaler = StandardScaler()
        y_train_norm = logT_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        model = SimpleNN(len(feature_cols), hidden_layers, hidden_neurons)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        X_tensor = torch.FloatTensor(X_train_norm)
        y_tensor = torch.FloatTensor(y_train_norm)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()

        def compute_rms(X, y_true, feat_scaler, logT_scaler, model):
            if len(X) == 0:
                return np.nan
            X_norm = feat_scaler.transform(X)
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_norm)
                y_pred_norm = model(X_tensor).numpy()
                y_pred = logT_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
            return np.sqrt(np.mean((y_pred - y_true) ** 2))

        train_rms = compute_rms(X_train, y_train, feat_scaler, logT_scaler, model)
        test_rms = compute_rms(X_test, y_test, feat_scaler, logT_scaler, model) if len(test_df) > 0 else np.nan
        n_train = len(train_df)
        n_test = len(test_df)
        if n_test > 0:
            overall_rms = np.sqrt(
                (n_train * train_rms ** 2 + n_test * test_rms ** 2) / (n_train + n_test)
            )
        else:
            overall_rms = train_rms

        return train_rms, test_rms, overall_rms

    print("[processing] Starting independent model trainings...")
    results = []
    start_time = time.time()

    for idx, subset in enumerate(top_subsets):
        k_val = subset['k']
        row_id = subset['row_id']
        random_seed = random_seed_base + row_id

        train_path = Path(model_1_6_dir) / f"train_k{k_val}_row{row_id}.csv"
        test_path = Path(model_1_6_dir) / f"test_k{k_val}_row{row_id}.csv"

        if not train_path.exists() or not test_path.exists():
            print(f"[check] Skipping k={k_val}, row_id={row_id}: missing files")
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if 'half_life_s_exp' in train_df.columns and 'half_life_s' not in train_df.columns:
            train_df['half_life_s'] = train_df['half_life_s_exp']
        if 'half_life_s_exp' in test_df.columns and 'half_life_s' not in test_df.columns:
            test_df['half_life_s'] = test_df['half_life_s_exp']

        co_train_rms, co_test_rms, co_overall_rms = train_and_evaluate_single_model(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            model_name=f"cluster_only_k{k_val}_row{row_id}",
            epochs=train_epochs,
            lr=1e-3,
            random_seed=random_seed,
            batch_size=batch_size
        )

        combined_train = pd.concat([train_df[feature_cols + ['half_life_s']], alpha_data_shared], ignore_index=True)
        comb_train_rms, comb_test_rms, comb_overall_rms = train_and_evaluate_single_model(
            train_df=combined_train,
            test_df=test_df,
            feature_cols=feature_cols,
            model_name=f"combined_k{k_val}_row{row_id}",
            epochs=train_epochs,
            lr=1e-3,
            random_seed=random_seed,
            batch_size=batch_size
        )

        results.append({
            'k': k_val,
            'row_id': row_id,
            'cluster_only_train_rms': co_train_rms,
            'cluster_only_test_rms': co_test_rms,
            'cluster_only_overall_rms': co_overall_rms,
            'combined_train_rms': comb_train_rms,
            'combined_test_rms': comb_test_rms,
            'combined_overall_rms': comb_overall_rms
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            est_total = elapsed / (idx + 1) * len(top_subsets)
            print(f"[processing] Progress: {idx + 1}/{len(top_subsets)} subsets completed "
                  f"({elapsed / 60:.1f} min elapsed, ~{est_total / 60:.1f} min total)")

    results_df = pd.DataFrame(results)
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"[output] Results saved to: {output_csv_path}")

    return results_df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_pretrained_model_with_dimension_match(state_dict: dict, current_input_dim: int = 5) -> dict:
    """
    Load pretrained model with automatic dimension matching for input layer.

    Parameters
    ----------
    state_dict : dict
        Pretrained model state dictionary
    current_input_dim : int, optional
        Current model input dimension (default: 5)

    Returns
    -------
    new_state_dict : dict
        Modified state dictionary with matched dimensions
    """
    first_layer_weight_key = None
    for key in state_dict.keys():
        if 'weight' in key and ('layers.0' in key or 'hidden_0' in key or 'layers.hidden_0' in key):
            first_layer_weight_key = key
            break

    if first_layer_weight_key is None:
        raise ValueError("Could not find first layer weight key in state_dict")

    first_layer_weight = state_dict[first_layer_weight_key]
    original_input_dim = first_layer_weight.shape[1]
    print(f"Detected original input dimension: {original_input_dim}, current: {current_input_dim}")

    if original_input_dim == current_input_dim:
        print("Input dimensions match. Loading model directly.")
        return state_dict

    if original_input_dim < current_input_dim:
        print(f"Expanding input dimension from {original_input_dim} to {current_input_dim}")
        new_state_dict = state_dict.copy()
        old_weight = state_dict[first_layer_weight_key]
        bias_key = first_layer_weight_key.replace('weight', 'bias')
        old_bias = state_dict[bias_key] if bias_key in state_dict else None
        new_features = current_input_dim - original_input_dim
        new_weight_part = torch.randn(old_weight.shape[0], new_features) * 0.01
        new_weight = torch.cat([old_weight, new_weight_part], dim=1)
        new_state_dict[first_layer_weight_key] = new_weight
        if bias_key and old_bias is not None:
            new_state_dict[bias_key] = old_bias
        return new_state_dict
    else:
        raise ValueError(f"Cannot reduce input dimension from {original_input_dim} to {current_input_dim}. "
                         "This would lose important information from the pre-trained model.")


def freeze_model_layers(model: nn.Module, num_frozen_hidden_layers: int):
    """
    Freeze specified number of hidden layers in the model.

    Parameters
    ----------
    model : nn.Module
        Neural network model
    num_frozen_hidden_layers : int
        Number of hidden layers to freeze
    """
    if num_frozen_hidden_layers >= 1:
        for param in model.layers[0].parameters():
            param.requires_grad = False
        for i in range(1, num_frozen_hidden_layers):
            if i * 2 < len(model.layers):
                for param in model.layers[i * 2].parameters():
                    param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params / total_params * 100:.2f}%)")


def get_16_subset_data(
        k: int,
        row_id: int,
        model_1_6_dir: str,
        feature_cols: List[str] = ['Z', 'A', 'Q_MeV', 'Z_k', 'A_k']
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load 1_6 model subset data from pre-saved CSV files.

    Parameters
    ----------
    k : int
        Number of clusters
    row_id : int
        Row ID of the subset
    model_1_6_dir : str
        Directory containing train/test CSV files
    feature_cols : list, optional
        Feature column names (default: ['Z', 'A', 'Q_MeV', 'Z_k', 'A_k'])

    Returns
    -------
    train_df : pd.DataFrame
        Training dataset
    test_df : pd.DataFrame
        Test dataset
    """
    model_dir = Path(model_1_6_dir)
    train_path = model_dir / f"train_k{k}_row{row_id}.csv"
    test_path = model_dir / f"test_k{k}_row{row_id}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    def adapt_half_life(df):
        df = df.copy()
        if 'half_life_s_exp' in df.columns:
            df['half_life_s'] = df['half_life_s_exp']
        elif 'half_life_s' not in df.columns:
            raise ValueError(
                f"Missing half-life column. Available: {list(df.columns)}\n"
                f"Expected 'half_life_s' or 'half_life_s_exp'"
            )
        return df

    train_df = adapt_half_life(train_df)
    test_df = adapt_half_life(test_df)

    missing_train = [col for col in feature_cols if col not in train_df.columns]
    missing_test = [col for col in feature_cols if col not in test_df.columns]
    if missing_train or missing_test:
        raise ValueError(
            f"Missing feature columns for k={k}, row_id={row_id}:\n"
            f"  Train ({len(train_df)} samples): {missing_train}\n"
            f"  Test ({len(test_df)} samples): {missing_test}"
        )

    if not hasattr(get_16_subset_data, '_first_loaded'):
        print(f"\n✓ First subset loaded successfully (k={k}, row_id={row_id})")
        print(f"  Directory: {model_dir}")
        print(f"  Train samples: {len(train_df)} | Test samples: {len(test_df)}")
        print(f"  Columns: {list(train_df.columns[:6])}...")
        get_16_subset_data._first_loaded = True

    return train_df, test_df


def _save_final_csv(
        all_results: list,
        selected_seeds: list,
        output_path: Path,
        group_size: int = 27
):
    """
    Save final CSV with selected seeds' predictions.

    Parameters
    ----------
    all_results : list
        List of all seed results
    selected_seeds : list
        List of selected seed IDs
    output_path : Path
        Output file path
    group_size : int, optional
        Number of rows to keep per seed (default: 27)
    """
    rows = []
    for item in all_results:
        seed = item['seed']
        if seed not in selected_seeds:
            continue
        df = pd.DataFrame(item['results_dict']['all_results'])
        df['seed'] = seed
        df['source_seed'] = seed
        n_keep = min(len(df), group_size)
        selected_rows = df.head(n_keep)
        for _, row in selected_rows.iterrows():
            rows.append(row.to_dict())

    if rows:
        final_df = pd.DataFrame(rows)
        if 'seed' in final_df.columns:
            cols = ['seed'] + [c for c in final_df.columns if c != 'seed']
            final_df = final_df[cols]
        final_df.to_csv(output_path, index=False)


def _print_seed_summary(all_results: list, selected_seeds: list, path_name: str):
    """
    Print RMS summary for selected seeds.

    Parameters
    ----------
    all_results : list
        List of all seed results
    selected_seeds : list
        List of selected seed IDs
    path_name : str
        Name of the selection path/method
    """
    print(f"\n{'=' * 80}")
    print(f"{path_name} SELECTED SEEDS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Selected seeds ({len(selected_seeds)}): {sorted(selected_seeds)}")

    selected_results = [item for item in all_results if item['seed'] in selected_seeds]
    if selected_results:
        avg_train_rms = np.mean([item['train_rms'] for item in selected_results])
        avg_test_rms = np.mean([item['test_rms'] for item in selected_results])
        avg_overall_rms = np.mean([item['overall_rms'] for item in selected_results])
        print(f"\nRMS Statistics (averaged over {len(selected_results)} seeds):")
        print(f"  Train RMS:  {avg_train_rms:.4f}")
        print(f"  Test RMS:   {avg_test_rms:.4f}")
        print(f"  Overall RMS: {avg_overall_rms:.4f}")
    print(f"{'=' * 80}")