"""
===============================================================================
PLOTTER MODULE - Scientific Visualization for Nuclear Decay Transfer Learning
===============================================================================

This module provides professional scientific plotting functions for visualizing
transfer learning results in nuclear alpha/cluster decay prediction tasks.

Key Features:
- Six-panel RMS comparison plots with cross-validation statistics
- Seed-based scatter residual plots with nuclide labeling
- Isotope scatter plots with UDL baseline comparison
- K-vs-RMS performance curves with uncertainty bands

Dependencies:
- matplotlib>=3.5.0
- pandas>=1.4.0
- numpy>=1.21.0

Author: [Your Name]
Date: 2026
"""

import os
import re
import warnings
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FixedLocator,
                               FormatStrFormatter, FuncFormatter)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm


# =============================================================================
# MAIN PLOTTING FUNCTIONS
# =============================================================================

def plot_neural_network_evaluation_barchart(
        fold_rms_alpha: str, fold_rms_cluster: str, fold_rms_combined: str,
        alpha_csv_path: str, cluster_only_csv_path: str, combined_csv_path: str,
        output_dir: str,
        alpha_combined_only: bool = True
) -> None:
    """
    Generate neural network evaluation bar charts showing RMS errors and residual distributions.
    """
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 32,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'Times New Roman',
        'axes.labelsize': 36,
        'axes.titlesize': 36,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        'legend.fontsize': 30,
        'figure.titlesize': 36
    })

    FS_Y, FS_T, FS_M, FS_X, FS_L, FS_P = 36, 32, 32, 32, 30, 36
    ROT_L, ROT_R = 30, 60

    if alpha_combined_only:
        decay_labels = [r'$\alpha$ decay model', r'$\alpha$ decay+cluster decay model']
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        fold_paths = [fold_rms_alpha, fold_rms_combined]
        res_paths = [alpha_csv_path, combined_csv_path]
        model_names = ['alpha', 'combined']
        n_rows = 2
        fig_size = (38, 19)
    else:
        decay_labels = [r'$\alpha$ decay', r'cluster decay model', r'$\alpha$ decay+cluster decay model']
        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        fold_paths = [fold_rms_alpha, fold_rms_cluster, fold_rms_combined]
        res_paths = [alpha_csv_path, cluster_only_csv_path, combined_csv_path]
        model_names = ['alpha', 'cluster_only', 'combined']
        n_rows = 3
        fig_size = (42, 22)

    c_tr, c_vs, c_ts = '#00C7C7', '#5E73FF', '#FFC212'
    c_tr_res, c_ts_res, c_mean = '#28A02D', '#FFB6C1', '#8B0000'

    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(n_rows, 2, hspace=0.08, wspace=0.08, width_ratios=[1, 1])
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(n_rows)]

    for row, fold_path in enumerate(fold_paths):
        ax = axes[row][0]
        ax.text(0.02, 0.98, f"{panel_labels[row * 2]} {decay_labels[row]}",
                transform=ax.transAxes, fontsize=FS_P, fontfamily='Times New Roman',
                fontweight='bold', va='top', ha='left')

        if not os.path.exists(fold_path):
            ax.text(0.5, 0.5, 'Unknown', ha='center', va='center',
                    transform=ax.transAxes, fontsize=32, color='gray')
            ax.set_ylabel(r'$\sigma_{RMS}$', fontsize=FS_Y, fontweight='bold', labelpad=10)
            ax.set_xticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(3.0)
            continue

        df = pd.read_csv(fold_path)
        tr_rms = pd.to_numeric(df.iloc[0], errors='coerce').values
        vs_rms = pd.to_numeric(df.iloc[1], errors='coerce').values
        ts_rms = pd.to_numeric(df.iloc[2], errors='coerce').values
        avg_tr, avg_vs, avg_ts = np.mean(tr_rms), np.mean(vs_rms), np.mean(ts_rms)

        x = np.arange(len(tr_rms))
        w = 0.25
        ax.bar(x - w, tr_rms, w, label='TR', color=c_tr, edgecolor='black',
               linewidth=3, alpha=0.66)
        ax.bar(x, vs_rms, w, label='VS', color=c_vs, edgecolor='black',
               linewidth=3, alpha=0.66)
        ax.bar(x + w, ts_rms, w, label='TS', color=c_ts, edgecolor='black',
               linewidth=3, alpha=0.66)

        ax.axhline(avg_tr, color=c_tr, linestyle='-.', linewidth=4, alpha=0.8,
                   label=r'$\overline{\sigma}^{TR}_{RMS}$')
        ax.axhline(avg_vs, color=c_vs, linestyle='-.', linewidth=4, alpha=0.8,
                   label=r'$\overline{\sigma}^{VS}_{RMS}$')
        ax.axhline(avg_ts, color=c_ts, linestyle='-.', linewidth=4, alpha=0.8,
                   label=r'$\overline{\sigma}^{TS}_{RMS}$')

        y_min, y_max = ax.get_ylim()
        if row == n_rows - 1:
            y_max *= 1.3
        ax.set_ylim(y_min, y_max)

        y_range = y_max - y_min
        if y_range < 2:
            major_ticks = list(range(int(np.floor(y_min)), int(np.ceil(y_max)) + 1))
        elif y_range < 5:
            major_ticks = list(range(int(np.floor(y_min)), int(np.ceil(y_max)) + 1))
        elif y_range < 10:
            major_ticks = list(range(int(np.floor(y_min / 2) * 2),
                                     int(np.ceil(y_max / 2) * 2) + 1, 2))
        else:
            major_ticks = list(range(int(np.floor(y_min / 5) * 5),
                                     int(np.ceil(y_max / 5) * 5) + 1, 5))

        ax.yaxis.set_major_locator(FixedLocator(sorted(list(set(major_ticks)))))

        x_min, x_max = ax.get_xlim()
        x_pos = x_max - (x_max - x_min) * 0.04
        ch = y_range * 0.05
        y_st = max(avg_tr, avg_vs, avg_ts) + ch * 0.3

        sorted_avgs = sorted([(avg_tr, c_tr, f'{avg_tr:.3f}'),
                              (avg_vs, c_vs, f'{avg_vs:.3f}'),
                              (avg_ts, c_ts, f'{avg_ts:.3f}')],
                             key=lambda x: x[0], reverse=True)
        y_top = y_st + (len(sorted_avgs) - 1) * ch * 1.5
        for i, (v, c, t) in enumerate(sorted_avgs):
            ax.text(x_pos, y_top - i * ch * 1.5, t, color=c, va='bottom', ha='right',
                    fontsize=FS_M, fontfamily='Times New Roman', fontweight='bold')

        ax.set_ylabel(r'$\sigma_{RMS}$', fontsize=FS_Y, fontweight='bold', labelpad=10)
        if row == n_rows - 1:
            ax.set_xticks(x)
            ax.set_xticklabels([f'Fold {i + 1}' for i in range(len(tr_rms))],
                               fontsize=FS_X, rotation=ROT_L, ha='right')
        else:
            ax.set_xticks([])

        ax.set_xlim(-0.5, len(tr_rms) - 0.5)
        for sp in ax.spines.values():
            sp.set_linewidth(3.0)
        for lb in ax.get_yticklabels():
            lb.set_fontsize(FS_T)
        ax.tick_params(axis='both', direction='in', which='both', width=3.0)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    for row, (csv_path, mn) in enumerate(zip(res_paths, model_names)):
        ax = axes[row][1]
        ax.text(0.02, 0.98, f"{panel_labels[row * 2 + 1]} {decay_labels[row]}",
                transform=ax.transAxes, fontsize=FS_P, fontfamily='Times New Roman',
                fontweight='bold', va='top', ha='left')

        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, f'{decay_labels[row]}\nData Not Found',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=24, color='gray')
            ax.set_ylabel(r'$|\log_{10}(T_{\mathrm{DNN}}/T_{\mathrm{EXP}})|$',
                          fontsize=FS_Y, fontweight='bold')
            continue

        try:
            df = pd.read_csv(csv_path)
            if mn == 'alpha':
                df = df[df['dataset_type'] == 'test'].copy()
                if len(df) == 0:
                    raise ValueError()

            df = df.sort_values(['N', 'Z']).reset_index(drop=True)
            df['lc'] = df.groupby('nucleus_label').cumcount() + 1
            mc = df.groupby('nucleus_label')['lc'].max()
            df['dl'] = df.apply(
                lambda r: f"{r['nucleus_label']}$^{{{r['lc']}}}$" if mc[r['nucleus_label']] > 1
                else r['nucleus_label'], axis=1)
            df['ip'] = np.abs(df['ratio']) < 1e-6
            df['res'] = np.abs(df['ratio'])
            df.loc[df['ip'], 'res'] = 1e-3
            df = df[np.isfinite(df['res']) & (df['res'] > 0)].copy()

            labels = df['dl'].values
            res = df['res'].values
            dt = df['dataset_type'].values
            ip = df['ip'].values
            x = np.arange(len(res))

            tr_m = dt == 'train'
            if np.any(tr_m):
                ax.bar(x[tr_m & ~ip], res[tr_m & ~ip], color=c_tr_res,
                       edgecolor='black', linewidth=3.0, alpha=0.78, width=0.78,
                       label='Training Set' if row == 0 else None)
                if np.any(tr_m & ip):
                    ax.bar(x[tr_m & ip], res[tr_m & ip], color=c_tr_res,
                           edgecolor='black', linewidth=3.0, alpha=0.78, width=0.78)

            ts_m = dt == 'test'
            if np.any(ts_m):
                ax.bar(x[ts_m], res[ts_m], color=c_ts_res, edgecolor='black',
                       linewidth=3.0, alpha=0.88, width=0.78,
                       label='Test Set' if row == 0 else None)

            ratios = df['ratio'].values
            rms_all = np.sqrt(np.mean(ratios ** 2))
            ax.axhline(rms_all, color=c_mean, linestyle='--', linewidth=4.0, alpha=0.88,
                       label=r'$\overline{|\log_{10}(T_{\mathrm{DNN}}/T_{\mathrm{EXP}})|}$' if row == 0 else None)

            max_res = np.max(res)
            y_max = max_res * 1.3
            ax.set_ylim(0, y_max)
            ax.set_xlim(-1, len(res))

            x_min, x_max = ax.get_xlim()
            x_pos = x_max - (x_max - x_min) * 0.01
            y_range = y_max
            y_pos = rms_all + y_range * 0.02
            ax.text(x_pos, y_pos, f'{rms_all:.3f}', color=c_mean, va='bottom', ha='right',
                    fontsize=FS_M, fontfamily='Times New Roman', fontweight='bold')

            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_ylabel(r'$|\log_{10}(T_{\mathrm{DNN}}/T_{\mathrm{EXP}})|$',
                          fontsize=FS_Y, fontweight='bold')

            if row == n_rows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=FS_X, rotation=ROT_R, ha='right')
                ax.tick_params(axis='x', direction='in', which='both', width=3.0,
                               top=False, bottom=True)
            else:
                ax.set_xticks([])

            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_linewidth(3.0)
            for lb in ax.get_yticklabels():
                lb.set_fontsize(FS_T)
            ax.tick_params(axis='y', direction='in', which='both', width=3.0,
                           left=True, right=False)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error\n{decay_labels[row]}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=18, color='red')
            ax.set_ylabel(r'$|\log_{10}(T_{\mathrm{DNN}}/T_{\mathrm{EXP}})|$',
                          fontsize=FS_Y, fontweight='bold')

    fig.align_ylabels([axes[i][0] for i in range(n_rows)])
    fig.align_ylabels([axes[i][1] for i in range(n_rows)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0.035, 1, 0.955])

    _old_mathtext_default = plt.rcParams.get('mathtext.default', 'it')
    plt.rcParams['mathtext.default'] = 'bf'

    lp = axes[0][0].get_position()
    hl, ll = axes[0][0].get_legend_handles_labels()
    if hl:
        fig.legend(hl, ll, fontsize=FS_L, loc='upper center',
                   bbox_to_anchor=(lp.x0 + lp.width / 2, lp.y1 + 0.048), ncol=6,
                   prop={'family': 'Times New Roman', 'size': FS_L, 'weight': 'bold'},
                   frameon=False, handlelength=2.0, handletextpad=0.6)

    rp = axes[0][1].get_position()
    handles_right = [
        Line2D([0], [0], color=c_mean, linestyle='--', linewidth=4.0, alpha=0.88),
        Patch(facecolor=c_tr_res, edgecolor='black', linewidth=3.0, alpha=0.78),
        Patch(facecolor=c_ts_res, edgecolor='black', linewidth=3.0, alpha=0.88)
    ]
    labels_right = [r'$\overline{|\log_{10}(T_{\mathrm{DNN}}/T_{\mathrm{EXP}})|}$', 'TR', 'TS']
    fig.legend(handles_right, labels_right, fontsize=FS_L, loc='upper center',
               bbox_to_anchor=(rp.x0 + rp.width / 2, rp.y1 + 0.050), ncol=3,
               prop={'family': 'Times New Roman', 'size': FS_L, 'weight': 'bold'},
               frameon=False, handlelength=2.0, handletextpad=0.6)

    plt.rcParams['mathtext.default'] = _old_mathtext_default

    for row in axes:
        for ax in row:
            if ax.get_legend():
                ax.legend().remove()

    os.makedirs(output_dir, exist_ok=True)
    output_filename = 'nn_evaluation_alpha_combined.png' if alpha_combined_only else 'nn_evaluation_full.png'
    plt.savefig(os.path.join(output_dir, output_filename),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


from matplotlib.ticker import FixedLocator, AutoMinorLocator
from matplotlib import cm
from pathlib import Path


def plot_seed_scatter_residuals(
        seed_results_path,
        output_dir: str,
        output_filename: str = "seed_scatter_residuals.png",
        mode: str = 'cluster',
        layout: str = 'up'
):
    """
    Generate scatter residual plots for seed-based model evaluation.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 32,
        'mathtext.fontset': 'stix',
        'axes.labelsize': 36,
        'axes.titlesize': 36,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        'legend.fontsize': 30,
        'figure.titlesize': 36
    })

    if mode not in ('cluster', 'alpha'):
        raise ValueError(f"mode must be 'cluster' or 'alpha'")

    if isinstance(seed_results_path, str):
        seed_results_path = [seed_results_path]

    results_dfs = []
    for path in seed_results_path:
        df = pd.read_csv(path).rename(columns=str.strip)
        if 'seed' not in df.columns:
            if 'alpha_seed' in df.columns:
                df['seed'] = df['alpha_seed']
            elif 'transfer_seed' in df.columns:
                df['seed'] = df['transfer_seed']
        results_dfs.append(df)

    C_TR = '#FFD663'
    C_TS = '#95A3FF'

    n_files = len(results_dfs)
    if mode == 'alpha':
        fig = plt.figure(figsize=(30, 9))
    elif n_files == 1:
        fig = plt.figure(figsize=(14, 9))
    else:
        if layout == 'left':
            fig = plt.figure(figsize=(28, 9))
        else:
            fig = plt.figure(figsize=(14, 16))

    def get_nucleus_label(n, z, particle, unique_data):
        A = n + z
        element_symbols = {
            56: 'Ba', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
            93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf',
            99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf'
        }
        symbol = element_symbols.get(int(z), 'Xx')
        same_nucleus = unique_data[(unique_data['N'] == n) & (unique_data['Z'] == z)]
        if len(same_nucleus) > 1:
            try:
                channel_idx = same_nucleus[same_nucleus['Emitted_Particle'] == particle].index[0]
                relative_idx = list(same_nucleus.index).index(channel_idx) + 1
                return f"$^{{{int(A)}}}${symbol}$^{{{relative_idx}}}$"
            except:
                return f"$^{{{int(A)}}}${symbol}"
        return f"$^{{{int(A)}}}${symbol}"

    def plot_single_ax(ax, results_df, mode):
        if mode == 'alpha':
            plot_df = results_df[results_df['dataset_type'] == 'test'].copy()
            if plot_df.empty:
                raise ValueError("No test data found for alpha mode.")

            unique_N = sorted(plot_df['N'].unique())
            n_pos_map = {n_val: idx for idx, n_val in enumerate(unique_N)}

            nuclide_colors = {}
            for n_val in unique_N:
                z_values = sorted(plot_df[plot_df['N'] == n_val]['Z'].unique())
                cmap = cm.get_cmap('tab10', max(len(z_values), 10))
                for idx, z_val in enumerate(z_values):
                    nuclide_colors[(int(n_val), int(z_val))] = cmap(idx % 10)

            for (n_val, z_val), group in plot_df.groupby(['N', 'Z']):
                ax.scatter([n_pos_map[int(n_val)]] * len(group),
                           group['ratio'].astype(float),
                           c=[nuclide_colors[(int(n_val), int(z_val))]],
                           marker='o', s=220, alpha=0.85, edgecolors='black', linewidth=3.0, zorder=3)

            ax.set_xticks(range(len(unique_N)))
            ax.set_xticklabels([str(int(n)) for n in unique_N], fontsize=30, rotation=45, ha='right')
            ax.set_xlabel('Neutron Number (N)', fontsize=36)
            ax.set_ylabel(r'$\log_{10}(T_{\mathrm{EXP}}/T_{\mathrm{Alpha}})$', fontsize=36)
            if len(unique_N) > 1:
                ax.fill_between([0, len(unique_N) - 1], -1.0, 1.0, color='#FF6363', alpha=0.1)

        else:
            unique_data = results_df.drop_duplicates(subset=['N', 'Z', 'Emitted_Particle']).copy()
            unique_data = unique_data.sort_values(['N', 'Z', 'Emitted_Particle']).reset_index(drop=True)
            unique_data['x_pos'] = range(len(unique_data))

            pos_map = {(row['N'], row['Z'], row['Emitted_Particle']): row['x_pos']
                       for _, row in unique_data.iterrows()}
            results_df['x_pos'] = results_df.apply(
                lambda row: pos_map[(row['N'], row['Z'], row['Emitted_Particle'])], axis=1)

            unique_data['nuclide_label'] = unique_data.apply(
                lambda row: get_nucleus_label(row['N'], row['Z'], row['Emitted_Particle'], unique_data), axis=1)

            tr_mask = (results_df['dataset_type'] == 'train') & (results_df['ratio'].abs() <= 1.5)
            ts_mask = results_df['dataset_type'] == 'test'

            ax.scatter(results_df.loc[tr_mask, 'x_pos'].astype(float),
                       results_df.loc[tr_mask, 'ratio'].astype(float),
                       c=C_TR, marker='o', s=200, alpha=0.8, edgecolors='black', linewidth=3.0,
                       label='Train', zorder=3)
            ax.scatter(results_df.loc[ts_mask, 'x_pos'].astype(float),
                       results_df.loc[ts_mask, 'ratio'].astype(float),
                       c=C_TS, marker='o', s=200, alpha=0.8, edgecolors='darkblue', linewidth=3.0,
                       label='Test', zorder=2)

            ax.set_xticks(range(len(unique_data)))
            ax.set_xticklabels(unique_data['nuclide_label'].values, fontsize=28, rotation=60, ha='right')
            ax.set_xlabel('', fontsize=36)
            ax.set_ylabel(r'$\log_{10}(T_{\mathrm{EXP}}/T_{\mathrm{Cluster}})$', fontsize=36)
            ax.fill_between([0, len(unique_data) - 1], -1.0, 1.0, color='#FF6363', alpha=0.1)

            # 修改：图例紧挨着 (a)/(b) 标签
            # loc='upper left' 表示 anchor 点是图例框的左上角
            # bbox_to_anchor=(0.1, 0.99) 将图例左上角定在 x=0.1, y=0.99 处
            # (a) 标签在 x=0.02，这样两者之间只有很小的间隙，实现“紧挨着”的效果
            ax.legend(loc='upper left',bbox_to_anchor=(-0.03, 1.02),
                      fontsize=30, frameon=True, fancybox=False, framealpha=0, edgecolor='black')

    def configure_ax(ax, show_xlabel=True, subplot_label=None):
        ax.axhline(0.0, color='k', linestyle='-', linewidth=3.0)
        ax.axhline(1.0, color='#FF6363', linestyle='-.', linewidth=3.0, alpha=0.7)
        ax.axhline(-1.0, color='#FF6363', linestyle='-.', linewidth=3.0, alpha=0.7)
        ax.set_ylim(-8, 8)
        ax.yaxis.set_major_locator(FixedLocator(np.array([-8., -4., 0., 4., 8.])))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis='both', which='major', direction='in', length=12, width=3.0,
                       labelsize=32, top=False, right=False, bottom=True, left=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=6, width=2.5,
                       top=False, right=False, bottom=True, left=True)
        for sp in ax.spines.values():
            sp.set_linewidth(3.0)

        # 控制横轴标签显示
        if not show_xlabel:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        # 子图标签 (a) 放在坐标轴内左上角最左侧
        if subplot_label is not None:
            ax.text(0.93, 0.98, f'({subplot_label})', transform=ax.transAxes,
                    fontsize=32, fontweight='bold', va='top', ha='left')

    # ========== 生成双子图 ==========
    axes_list = []
    if n_files == 1:
        ax_r = fig.add_subplot(111)
        plot_single_ax(ax_r, results_dfs[0], mode)
        axes_list.append((ax_r, True, 'a'))
    else:
        for i, df in enumerate(results_dfs):
            if layout == 'left':
                ax = fig.add_subplot(1, n_files, i + 1)
            else:
                ax = fig.add_subplot(n_files, 1, i + 1)
            plot_single_ax(ax, df, mode)
            # 只有最后一个子图显示横轴标签
            show_xlabel = (i == n_files - 1)
            # 子图标签：a, b, c, ...
            subplot_label = chr(ord('a') + i)
            axes_list.append((ax, show_xlabel, subplot_label))

    for ax, show_xlabel, subplot_label in axes_list:
        configure_ax(ax, show_xlabel=show_xlabel, subplot_label=subplot_label)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / output_filename

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[output] Seed scatter plot ({mode}, {n_files} file(s), layout={layout}) saved to: {output_path}")
    plt.close()

    # ========== 传入多个CSV时，分别保存各自单图 ==========
    if n_files > 1:
        for i, df in enumerate(results_dfs):
            # 获取原始CSV文件名作为单图文件名的一部分
            if isinstance(seed_results_path, list):
                csv_name = Path(seed_results_path[i]).stem
            else:
                csv_name = f"file_{i + 1}"

            # 生成单图文件名
            single_output_filename = f"{Path(output_filename).stem}_{csv_name}{Path(output_filename).suffix}"
            single_output_path = output_dir_path / single_output_filename

            # 创建单图（规格与n_files==1时一致）
            if mode == 'alpha':
                single_fig = plt.figure(figsize=(30, 9))
            else:
                single_fig = plt.figure(figsize=(14, 9))

            single_ax = single_fig.add_subplot(111)
            plot_single_ax(single_ax, df, mode)
            configure_ax(single_ax, show_xlabel=True, subplot_label=chr(ord('a') + i))

            plt.tight_layout()
            plt.savefig(str(single_output_path), dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[output] Single seed scatter plot ({mode}, file {i + 1}) saved to: {single_output_path}")
            plt.close(single_fig)


def plot_seed_rms_comparison(
        csv_path: str,
        output_dir: str,
        output_filename: str = "seed_rms_comparison.png",
        reference_seed: Optional[int] = None,
        top_n: int = 40,
        figsize: Tuple[int, int] = (8, 6),
        dpi: int = 300,
        rms_metric: str = 'overall',
        comparison_csv_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate bar chart comparing RMS errors for most similar seeds.
    Single-panel plot with cluster decay initialization evaluation style.
    """
    # ============================================== 导入依赖 ==============================================
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from matplotlib.ticker import MultipleLocator
    from typing import Optional, Tuple

    # ============================================== 样式设置 ==============================================
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 36,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'Times New Roman',
        'xtick.labelsize': 35,
        'ytick.labelsize': 35,
        'legend.fontsize': 30,
        'axes.labelsize': 32,
        'axes.titlesize': 35
    })

    # 颜色方案
    C_TEST = '#FF6B6B'
    C_OVERALL = '#4ECDC4'

    # ============================================== RMS 计算函数 ==============================================
    def calculate_seed_rms(df: pd.DataFrame, seed_col: str = 'seed') -> dict:
        results = {}
        for seed in df[seed_col].unique():
            seed_data = df[df[seed_col] == seed].copy()
            valid_ratio = seed_data['ratio'].dropna()
            valid_ratio = valid_ratio[np.isfinite(valid_ratio)]
            if len(valid_ratio) == 0:
                continue
            overall_rms = np.sqrt(np.mean(valid_ratio ** 2))
            test_data = seed_data[seed_data['dataset_type'] == 'test']
            test_ratio = test_data['ratio'].dropna()
            test_ratio = test_ratio[np.isfinite(test_ratio)]
            if len(test_ratio) > 0:
                test_rms = np.sqrt(np.mean(test_ratio ** 2))
            else:
                test_rms = np.nan
            results[int(seed)] = {'test_rms': test_rms, 'overall_rms': overall_rms}
        return results

    # ============================================== 选择相似种子 ==============================================
    def get_top_seeds(rms_dict: dict, reference_seed: Optional[int],
                      top_n: int, rms_metric: str, file_label: str = "") -> list:
        if reference_seed is not None and reference_seed in rms_dict:
            ref_value = rms_dict[reference_seed][f'{rms_metric}_rms']
            if np.isnan(ref_value):
                reference_seed = None
        if reference_seed is not None and reference_seed in rms_dict:
            ref_value = rms_dict[reference_seed][f'{rms_metric}_rms']
            distances = []
            for seed, values in rms_dict.items():
                val = values[f'{rms_metric}_rms']
                if np.isnan(val):
                    continue
                dist = abs(val - ref_value)
                distances.append((seed, dist))
            distances.sort(key=lambda x: x[1])
            top_seeds = [seed for seed, _ in distances[:top_n]]
        else:
            seed_rms_list = []
            for seed, values in rms_dict.items():
                val = values[f'{rms_metric}_rms']
                if not np.isnan(val):
                    seed_rms_list.append((seed, val))
            seed_rms_list.sort(key=lambda x: x[1])
            top_seeds = [seed for seed, _ in seed_rms_list[:top_n]]
        if file_label:
            print(f"[info] {file_label}: Selected {len(top_seeds)} seeds (metric: {rms_metric}_rms)")
        return top_seeds

    # ============================================== 处理第二个 CSV 文件（只输出不绘图）==============================================
    if comparison_csv_path is not None:
        print(f"\n{'=' * 60}")
        print(f"[comparison] Processing comparison CSV: {comparison_csv_path}")
        print(f"{'=' * 60}\n")
        df_comp = pd.read_csv(comparison_csv_path)
        rms_dict_comp = calculate_seed_rms(df_comp)
        if not rms_dict_comp:
            raise ValueError("No valid RMS data found in comparison CSV file")
        top_seeds_comp = get_top_seeds(rms_dict_comp, reference_seed, top_n, rms_metric,
                                       file_label=f"Comparison ({Path(comparison_csv_path).name})")
        test_rms_vals = [rms_dict_comp[seed]['test_rms'] for seed in top_seeds_comp
                         if not np.isnan(rms_dict_comp[seed]['test_rms'])]
        overall_rms_vals = [rms_dict_comp[seed]['overall_rms'] for seed in top_seeds_comp
                            if not np.isnan(rms_dict_comp[seed]['overall_rms'])]
        avg_test_rms = np.mean(test_rms_vals) if test_rms_vals else float('nan')
        avg_overall_rms = np.mean(overall_rms_vals) if overall_rms_vals else float('nan')
        print(f"\n{'=' * 60}")
        print(f"Transferlearning Seeds evaluation - Average RMS ")
        print(f"{'=' * 60}")
        print(f"Average Test RMS:    {avg_test_rms:.4f}")
        print(f"Average Overall RMS: {avg_overall_rms:.4f}")
        print(f"{'=' * 60}\n")

    # ============================================== 数据加载（主 CSV 文件）==============================================
    print(f"[loading] Reading CSV file...")
    df = pd.read_csv(csv_path)
    rms_dict = calculate_seed_rms(df)
    if not rms_dict:
        raise ValueError("No valid RMS data found in CSV file")
    top_seeds = get_top_seeds(rms_dict, reference_seed, top_n, rms_metric,
                              file_label=f"Main ({Path(csv_path).name})")

    # ============================================== 提取绘图数据 ==============================================
    def extract_plot_data(rms_dict: dict, seed_list: list) -> Tuple[list, list, list]:
        seeds, test_vals, overall_vals = [], [], []
        for seed in seed_list:
            if seed in rms_dict:
                seeds.append(str(seed))
                test_vals.append(rms_dict[seed]['test_rms'])
                overall_vals.append(rms_dict[seed]['overall_rms'])
        return seeds, test_vals, overall_vals

    seeds, test_rms, overall_rms = extract_plot_data(rms_dict, top_seeds)
    if len(seeds) == 0:
        raise ValueError("No valid data extracted for plotting")
    seed_labels = [f'seed{i + 1}' for i in range(len(seeds))]

    # ============================================== 创建图形 ==============================================
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.arange(len(seeds))
    width = 0.35

    bars_test = ax.bar(x - width / 2, test_rms, width, label='Test RMS',
                       color=C_TEST, edgecolor='black', linewidth=1.5, alpha=0.85)
    bars_overall = ax.bar(x + width / 2, overall_rms, width, label='Overall RMS',
                          color=C_OVERALL, edgecolor='black', linewidth=1.5, alpha=0.85)

    ax.set_ylabel(r'$\bar{\sigma}_{\mathrm{RMS}}$', fontsize=40, fontweight='bold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels, rotation=45, ha='right', fontsize=26)
    ax.grid(axis='y', linestyle='--', alpha=0.25, linewidth=0.8)
    ax.set_ylim(0, 2.5)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    # 【关键修复1】设置x轴范围，缩小左右边距
    ax.set_xlim(-0.5, len(seeds) - 0.5)

    ax.tick_params(axis='both', which='both', direction='in', labelsize=32, width=2.0, length=6,
                   top=True, right=True)
    ax.tick_params(axis='both', which='minor', length=4, width=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)

    # 【关键修复2】图例移到左上角
    ax.legend(loc='upper left', frameon=False, fontsize=32, ncol=2,
              handletextpad=0.5, labelspacing=0.3, columnspacing=1.0, handlelength=1.5)

    # ============================================== 保存与输出 ==============================================
    plt.tight_layout(pad=1.8)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if not output_filename.lower().endswith('.png'):
        output_filename = output_filename + '.png'

    output_path = output_dir_path / output_filename

    plt.savefig(str(output_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    if output_path.exists():
        print(f"[output] Seed RMS comparison plot saved successfully to: {output_path}")
    else:
        print(f"[error] Failed to save plot to: {output_path}")
        raise RuntimeError(f"Failed to save plot to {output_path}")

    return fig


def plot_isotope_scatter_combined(
        k10_csv_path: str, full_csv_path: str, udl_csv_path: str,
        output_path: str, plot_config: dict, plot_full: bool = False) -> bool:
    """
    Generate isotope scatter plots showing experimental limits vs multi-model predictions.

    Parameters
    ----------
    k10_csv_path : str
        Path to K=10 model prediction results CSV
    full_csv_path : str
        Path to full model prediction results CSV
    udl_csv_path : str
        Path to UDL model prediction results CSV
    output_path : str
        Full output path for saved figure
    plot_config : dict
        Plot configuration dictionary containing colors, sizes, etc.
    plot_full : bool, optional
        Whether to display full model prediction points (default: False)

    Returns
    -------
    bool
        True if plotting succeeded
    """
    try:
        for p in [k10_csv_path, full_csv_path, udl_csv_path]:
            if not Path(p).exists():
                return False

        def proc(df, dnn=True):
            if 'Z' in df.columns and 'A' in df.columns:
                df = df[~((df['Z'] == 56) & (df['A'] == 114))].copy()
            if dnn and 'certainty' in df.columns:
                df = df[df['certainty'] == False].copy()
            if 'nucleus_label' in df.columns and 'Emitted_Particle' in df.columns:
                df = df.drop_duplicates(subset=['nucleus_label', 'Emitted_Particle'], keep='first').reset_index(
                    drop=True)
            if 'A' in df.columns:
                df = df.sort_values('A').reset_index(drop=True)
            return df

        df_k10, df_full, df_udl = proc(pd.read_csv(k10_csv_path)), proc(pd.read_csv(full_csv_path)), proc(
            pd.read_csv(udl_csv_path), False)

        def decay_id(df):
            if 'nucleus_label' in df.columns and 'Emitted_Particle' in df.columns:
                cl = lambda p: str(p).replace('$', '').replace(r'\mathrm{', '').replace('}', '').strip()
                ms = lambda c: (re.search(r'^(\d+)[A-Za-z]+', str(c).strip()).group(1) if re.search(r'^(\d+)[A-Za-z]+',
                                                                                                    str(c).strip()) else str(
                    c).strip()) if not pd.isna(c) else 'X'
                return df['nucleus_label'].apply(cl) + '_' + df['Emitted_Particle'].apply(ms)
            return df['Z'].astype(str) + '-' + df['A'].astype(str) if 'Z' in df.columns else df.index.astype(str)

        df_k10['did'], df_full['did'], df_udl['did'] = decay_id(df_k10), decay_id(df_full), decay_id(df_udl)
        if 'logT_pred_udl' not in df_udl.columns:
            return False

        df_k10 = df_k10.merge(df_udl[['did', 'logT_pred_udl']], on='did', how='left')
        df_full = df_full.merge(df_udl[['did', 'logT_pred_udl']], on='did', how='left')
        df_k10['src'], df_full['src'] = 'k10', 'full'
        df_comb = pd.concat([df_k10, df_full], ignore_index=True)

        uniq = pd.concat(
            [df_k10[['did', 'A', 'nucleus_label', 'logT_exp', 'Emitted_Particle', 'Z']].drop_duplicates('did'),
             df_full[['did', 'A', 'nucleus_label', 'logT_exp', 'Emitted_Particle', 'Z']].drop_duplicates('did')
             ]).drop_duplicates('did').sort_values('A').reset_index(drop=True)
        uniq['xb'] = np.arange(len(uniq))

        def ext_nuc(l):
            a = re.search(r'\^\{(\d+)\}', str(l))
            e = re.search(r'([A-Z][a-z]*)$', str(l).replace('$', '').replace(r'\mathrm{', '').replace('}', '').strip())
            return (a.group(1) if a else '', e.group(1) if e else '')

        def ext_em(em):
            m = re.match(r'^(\d+)([A-Za-z]+)', str(em).strip())
            return (m.group(1), m.group(2)) if m else ('', '')

        uniq[['mn', 'es']] = uniq['nucleus_label'].apply(lambda x: pd.Series(ext_nuc(x)))
        uniq[['em', 'ee']] = uniq['Emitted_Particle'].apply(lambda x: pd.Series(ext_em(x)))
        df_comb = df_comb.merge(uniq[['did', 'xb']], on='did', how='left')
        df_comb['xp'] = df_comb['xb']

        df_exp = uniq[['did', 'xb', 'logT_exp', 'mn', 'es', 'em', 'ee']].copy()
        df_exp['T_exp'] = 10 ** df_exp['logT_exp']
        df_udl_p = uniq[['did', 'xb']].copy()
        df_udl_p['logT_pred_udl'] = df_udl_p['did'].map(df_udl.set_index('did')['logT_pred_udl'].to_dict())
        df_udl_p['T_pred_udl'] = 10 ** df_udl_p['logT_pred_udl']

        plt.rcParams['font.family'] = plot_config['font_family']
        plt.rcParams['mathtext.fontset'] = 'stix'
        fig_w = plot_config['figsize'][0] * 1.15 * 0.55
        fig_h = plot_config['figsize'][1] * plot_config.get('y_shrink', 0.70)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=plot_config['dpi'])
        ax.set_yscale('log')
        SM = 1.5

        # Plot scatter points
        vu = df_udl_p['T_pred_udl'].notna()
        if vu.any():
            ax.scatter(df_udl_p.loc[vu, 'xb'], df_udl_p.loc[vu, 'T_pred_udl'],
                       c=plot_config['colors']['udl_prediction'],
                       s=plot_config['sizes']['predicted'] * SM, edgecolor='black', linewidth=1.0, zorder=3, marker='o')
        if plot_full:
            dfp = df_comb[df_comb['src'] == 'full'].copy()
            if len(dfp) > 0 and 'logT_pred' in dfp.columns:
                dfp['T_pred'] = 10 ** dfp['logT_pred']
                ax.scatter(dfp['xp'], dfp['T_pred'], facecolors='none', edgecolors='#FF6B6B',
                           s=plot_config['sizes']['predicted'] * 1.2 * SM, linewidth=2.0, zorder=3, marker='s')
        dfk = df_comb[df_comb['src'] == 'k10'].copy()
        if len(dfk) > 0 and 'logT_pred' in dfk.columns:
            dfk['T_pred'] = 10 ** dfk['logT_pred']
            ax.scatter(dfk['xp'], dfk['T_pred'], c='#FF6B6B', s=plot_config['sizes']['predicted'] * SM,
                       edgecolors='black', linewidth=1.0, zorder=3 if plot_full else 4, marker='o')
        ax.scatter(df_exp['xb'], df_exp['T_exp'], s=plot_config['sizes']['experimental'] * SM,
                   c='black', edgecolors='black', linewidth=1.0, marker='^', zorder=5)

        ax.set_ylabel(r'$\log_{10}(T\,/\,\mathrm{s})$', fontsize=20, labelpad=10)
        ax.set_ylim(float(10 ** 17.5), float(10 ** 35))

        # Y-axis tick settings
        major_ticks = [20, 25, 30, 35]
        ax.yaxis.set_major_locator(FixedLocator([float(10 ** i) for i in major_ticks]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{int(np.log10(y))}' if y > 0 else ''))
        ax.yaxis.set_minor_locator(FixedLocator([float(10 ** (i + 2.5)) for i in [17.5, 20, 25, 30]]))

        TW = plot_config['line_width']['axis']
        ax.tick_params(axis='both', which='major', direction='in', length=6.0, width=TW, top=True, right=True,
                       labelsize=20)
        ax.tick_params(axis='y', which='minor', direction='in', length=3.0, width=TW, left=True, right=True)
        ax.set_xticks(df_exp['xb'].values)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='major', direction='in', length=6.0, width=TW, top=True, bottom=True)
        xmin, xmax = float(df_exp['xb'].min()), float(df_exp['xb'].max())
        ax.set_xlim(xmin - 0.5 - plot_config.get('x_left_pad', 0.1), xmax + 0.5)

        # Add nuclide labels
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        skip = max(1, (len(df_exp) + 24) // 25)
        xo = plot_config.get('x_label_offset', 0.0)

        parent_label_y = plot_config.get('parent_label_y_axes', -0.02)
        emitted_label_y = plot_config.get('emitted_label_y_axes', 0.02)
        parent_fontsize = 20
        emitted_fontsize = parent_fontsize - 3

        for i in range(0, len(df_exp), skip):
            r = df_exp.iloc[i]
            xl = float(r['xb']) + xo
            ax.text(xl, parent_label_y, f"$^{{{r['mn']}}}${r['es']}",
                    transform=trans, fontsize=parent_fontsize, ha='right', va='top',
                    rotation=45, rotation_mode='anchor', clip_on=False)
            ax.text(xl, emitted_label_y, f"$^{{{r['em']}}}${r['ee']}",
                    transform=trans, fontsize=emitted_fontsize, ha='left', va='bottom',
                    rotation=45, rotation_mode='anchor', clip_on=False)

        for sp in ax.spines.values():
            sp.set_linewidth(TW)

        # Legend
        ms_u = math.sqrt(plot_config['sizes']['predicted'] * SM)
        ms_p = math.sqrt(plot_config['sizes']['predicted'] * SM)
        ms_e = math.sqrt(plot_config['sizes']['experimental'] * SM)

        handles = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markeredgecolor='black',
                   markersize=ms_e, markeredgewidth=1.0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plot_config['colors']['udl_prediction'],
                   markeredgecolor='black', markersize=ms_u, markeredgewidth=1.0),
        ]
        labels = ['low experimental limit', 'UDL']

        if plot_full:
            handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                                  markeredgecolor='#FF6B6B', markersize=ms_p * 1.1, markeredgewidth=2.0))
            labels.append('TL (full)')

        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markeredgecolor='black',
                              markersize=ms_p, markeredgewidth=1.0))
        labels.append('TL')

        ax.legend(handles, labels, loc='upper left', frameon=True, framealpha=0, fontsize=16,
                  handletextpad=0.3, handlelength=1.2, borderpad=0.3, labelspacing=0.2)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.96)
        plt.savefig(output_path, dpi=plot_config['dpi'], bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        return True
    except:
        return False


def plot_k_vs_rms_shared_subsets(
        base_model: Tuple[int, int], compare_models: List[Tuple[int, int]],
        model_labels: Dict[Tuple[int, int], str], model_colors: Dict[Tuple[int, int], str],
        best_n: int, udl_csv_path: str, transfer_results_dir: str, output_path: str,
        title: str = "", show_uncertainty: bool = True, figsize: Tuple[int, int] = (8, 6),
        dpi: int = 300, extrapolation_rms_path: Optional[str] = None,
        show_cluster_extrapolation: bool = False) -> bool:
    """
    Generate K-vs-RMS performance curve with uncertainty bands.
    """
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 22,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'Times New Roman',
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 18
    })

    try:
        udl_rms = pd.read_csv(udl_csv_path)['overall_sigma_rms'].min()
    except:
        udl_rms = 1.5

    res = {}
    for m in compare_models:
        cp = os.path.join(transfer_results_dir, f"model_{m[0]}_{m[1]}", "transfer_rms_summary.csv")
        if not os.path.exists(cp):
            continue
        df = pd.read_csv(cp)
        if 'k' not in df.columns and 'clusters' in df.columns:
            df['k'] = df['clusters']
        if 'overall_sigma_rms' not in df.columns and 'rms' in df.columns:
            df['overall_sigma_rms'] = df['rms']
        kv, mn, st = [], [], []
        for k in range(3, 11):
            kd = df[df['k'] == k]
            if kd.empty:
                continue
            if len(kd) >= best_n:
                b = kd.nsmallest(best_n, 'overall_sigma_rms')
                mn.append(b['overall_sigma_rms'].mean())
                st.append(b['overall_sigma_rms'].std(ddof=1))
            else:
                mn.append(kd['overall_sigma_rms'].mean())
                st.append(kd['overall_sigma_rms'].std(ddof=1) if len(kd) > 1 else 0)
            kv.append(k)
        if kv:
            res[m] = {'k': kv, 'mean_rms': mn, 'std_rms': st, 'min_rms': min(mn)}

    if extrapolation_rms_path and os.path.exists(extrapolation_rms_path):
        try:
            ed = pd.read_csv(extrapolation_rms_path)
            if show_cluster_extrapolation:
                for mt, col in [('cluster_only', 'cluster_only_overall_rms'), ('combined', 'combined_overall_rms')]:
                    kv, mn, st = [], [], []
                    for k in range(3, 11):
                        d = ed[ed['k'] == k][col].dropna().values
                        if len(d) > 0:
                            kv.append(k)
                            mn.append(np.mean(d))
                            st.append(np.std(d, ddof=1) if len(d) > 1 else 0)
                    if kv:
                        res[f"extrapolation_{mt}"] = {'k': kv, 'mean_rms': mn, 'std_rms': st, 'min_rms': min(mn)}
            else:
                mt, col = 'combined', 'combined_overall_rms'
                kv, mn, st = [], [], []
                for k in range(3, 11):
                    d = ed[ed['k'] == k][col].dropna().values
                    if len(d) > 0:
                        kv.append(k)
                        mn.append(np.mean(d))
                        st.append(np.std(d, ddof=1) if len(d) > 1 else 0)
                if kv:
                    res[f"extrapolation_{mt}"] = {'k': kv, 'mean_rms': mn, 'std_rms': st, 'min_rms': min(mn)}
        except:
            pass

    if not res:
        raise ValueError("No valid model data")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(2.85, 10.15)
    ax.set_ylim(0, 7.5)
    ax.axhline(y=udl_rms, color='#d62728', linewidth=2.0, linestyle='--', alpha=0.8, label='UDL', zorder=0)

    order = []
    if show_cluster_extrapolation and "extrapolation_combined" in res:
        order.append("extrapolation_combined")
    if show_cluster_extrapolation and "extrapolation_cluster_only" in res:
        order.append("extrapolation_cluster_only")
    if not show_cluster_extrapolation and "extrapolation_combined" in res:
        order.append("extrapolation_combined")
    for m in [(2, 6), (1, 6)]:
        if m in compare_models and m in res:
            order.append(m)
    for m in compare_models:
        if m not in [(1, 6), (2, 6)] and m in res:
            order.append(m)

    for key in order:
        r = res[key]
        if key == "extrapolation_combined":
            c, l, mk = '#2ca02c', r'DNN$_{\alpha + \mathrm{cluster}}$', '^'
        elif key == "extrapolation_cluster_only":
            c, l, mk = '#d62728', r'DNN$_{\mathrm{cluster}}$', 'v'
        elif key == (1, 6):
            c, l, mk = model_colors.get(key, '#1f77b4'), 'full fine-tuning', 'o'
        elif key == (2, 6):
            c, l, mk = model_colors.get(key, '#ff7f0e'), 'shallow fine-tuning', 's'
        else:
            c, l, mk = model_colors.get(key, '#000000'), model_labels.get(key, f"Model {key}"), 'o'
        sd = r['std_rms'] if show_uncertainty else [0] * len(r['std_rms'])
        if any(s > 1e-6 for s in sd):
            ax.fill_between(r['k'], np.array(r['mean_rms']) - np.array(sd), np.array(r['mean_rms']) + np.array(sd),
                            color=c, alpha=0.25, zorder=1)
        ax.plot(r['k'], r['mean_rms'], color=c, linewidth=2.0, marker=mk, markersize=8,
                markerfacecolor='white', markeredgecolor=c, markeredgewidth=1.5, label=l, zorder=2)

    ax.set_xlabel('Number of cluster decay samples in training set', fontsize=22)
    ax.set_ylabel(r'$\bar{\sigma}_{\mathrm{RMS}} \pm 1 \text{ SD}$', fontsize=22)
    ax.set_xticks(range(3, 11))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='both', which='both', direction='in', labelsize=22, width=1.8, length=5, top=True, right=True)
    ax.tick_params(axis='both', which='minor', length=3, width=1.2)
    for sp in ax.spines.values():
        sp.set_linewidth(1.8)

    h, l = ax.get_legend_handles_labels()
    if show_cluster_extrapolation:
        od = ['UDL', r'DNN$_{\mathrm{cluster}}$', r'DNN$_{\alpha + \mathrm{cluster}}$', 'full fine-tuning', 'shallow fine-tuning']
    else:
        od = ['UDL', r'DNN$_{\alpha + \mathrm{cluster}}$', 'full fine-tuning', 'shallow fine-tuning']
    oh, ol = [], []
    for t in od:
        for hi, li in zip(h, l):
            if li == t:
                oh.append(hi)
                ol.append(li)
                break
    ax.legend(oh, ol, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True, framealpha=0,
              fontsize=18, ncol=2, handletextpad=0.5, borderpad=0.4, labelspacing=0.3, columnspacing=1.0,
              handlelength=1.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=1.8)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_nucleus_label_by_n_logt(n_val: float, logt_val: float, cluster_data_path: str) -> str:
    """
    Generate nuclide label string from neutron number and log half-life.
    """
    element_symbols = {
        87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
        93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf',
        99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf'
    }

    cluster_df = pd.read_csv(cluster_data_path)
    cluster_df['logT_calc'] = np.log10(cluster_df['half_life_s'])

    has_certainty = 'certainty' in cluster_df.columns

    if has_certainty:
        cluster_df['is_certain'] = cluster_df['certainty'].astype(str).str.lower() == 'true'
        certain_matches = cluster_df[
            (np.abs(cluster_df['N'] - n_val) < 0.5) &
            cluster_df['is_certain']
            ].copy()

        if len(certain_matches) > 0:
            certain_matches['diff'] = np.abs(certain_matches['logT_calc'] - logt_val)
            best_match = certain_matches.loc[certain_matches['diff'].idxmin()]
            z = int(best_match['Z'])
            a = int(best_match['A'])
            symbol = element_symbols.get(z, '?')
            return f"$^{{{a}}}${symbol}"

    matches = cluster_df[np.abs(cluster_df['N'] - n_val) < 0.5].copy()
    if len(matches) == 0:
        return None

    matches['diff'] = np.abs(matches['logT_calc'] - logt_val)
    best_match = matches.loc[matches['diff'].idxmin()]

    z = int(best_match['Z'])
    a = int(best_match['A'])
    symbol = element_symbols.get(z, '?')
    return f"$^{{{a}}}${symbol}"


def _process_transfer_learning_data(df: pd.DataFrame) -> dict:
    """
    Process transfer learning RMS data: compute mean ± std for each k value.
    """
    k_values = []
    means = []
    stds = []

    for k_val in range(3, 11):
        k_df = df[df['k'] == k_val].copy()
        if k_df.empty:
            continue

        rms_vals = k_df['overall_sigma_rms'].dropna().values
        if len(rms_vals) == 0:
            continue

        k_values.append(k_val)
        means.append(np.mean(rms_vals))
        stds.append(np.std(rms_vals) if len(rms_vals) > 1 else 0)

    return {
        'k': k_values,
        'mean_rms': means,
        'std_rms': stds
    }