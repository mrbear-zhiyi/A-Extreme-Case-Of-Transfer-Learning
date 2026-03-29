from pathlib import Path
ROOT_DIR = Path(__file__).parent
"""
================================================
用户可配置外层形参
================================================
1、json文件路径
2、alpha与cluster数据文件路径
3、迁移学习输出目录
4、可视化输出目录


================================================
用户谨慎配置内层形参
================================================
1、Cluster映射关系表
2、训练模型输出目录
3、训练模型超参数配置
4、训练模型架构配置


================================================
相关配置辅助函数
================================================
1、路径获取函数
2、训练配置函数
3、路径获取函数
4、可视化配置函数
"""



"""
================================================用户可配置外层形参================================================
"""
# 数据路径
DATA_DIR = ROOT_DIR / "Data"
JSON_FILE_PATH = ROOT_DIR / "nuclear_decay_data_all.json"
ALPHA_DATA_PATH = DATA_DIR / "alpha_data_modified.csv"
CLUSTER_DATA_PATH = DATA_DIR / "cluster_data_modified.csv"
CLUSTER_DATA_EXPLORATION_PATH = DATA_DIR / "cluster_data_exploration.csv"
# 结果目录
RESULTS_DIR = ROOT_DIR / "results"
TRANSFER_RESULTS_DIR = RESULTS_DIR / "02_TransferLearning"
VISUALIZATION_DIR = RESULTS_DIR / "03_visualization"
PREDICTION_RESULTS_DIR = RESULTS_DIR / "04_Prediction"


"""
================================================用户谨慎配置内层形参================================================
"""


"""
=================================================cluster映射关系表=================================================
"""
PARTICLE_MAP = {
    'alpha': (2, 4), '4He': (2, 4), '14C': (6, 14), '20O': (8, 20),
    '23F': (9, 23), '22Ne': (10, 22), '24Ne': (10, 24), '25Ne': (10, 25),
    '26Ne': (10, 26), '28Mg': (12, 28), '30Mg': (12, 30),
    '32Si': (14, 32), '34Si': (14, 34)
}

"""
=================================================训练模型相关路径配置=================================================
"""

# 训练模型输出目录
NN_RESULTS_DIR = RESULTS_DIR / "01_NeuralNetwork"
LINEAR_DIR = TRANSFER_RESULTS_DIR / "Linear_fit"
UDL_CSV_PATH = LINEAR_DIR / "UDL comparison RMS.csv"
# 训练模型超参数配置
DEFAULT_HPARAMS = {
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'lambda_init': 1e-3,
    'max_epochs': 200,
    'patience': 20,
    'k_folds': 10,
}
# 训练模型架构配置 - Alpha
# ALPHA_LAYER_CONFIGS = {
#     1: [4, 5, 6, 10, 15],
#     2: [2, 3, 4, 5, 6, 10, 15],
#     3: [2, 3, 4, 5, 6, 10, 15],
#     4: [2, 3, 4, 5, 6, 10, 15]
# }
ALPHA_LAYER_CONFIGS = {
    1: [6],
}
# 训练模型架构配置 - Cluster
CLUSTER_LAYER_CONFIGS = {
    1: [6]
}
# 训练模型架构配置 - Combined
COMBINED_LAYER_CONFIGS = {
    1: [6]
}
#种子输出路径配置
MULTI_SEED_OUTPUT_DIR = NN_RESULTS_DIR / "Seed_evaluation"
# 评估模型架构配置
EVALUATE_LAYER_CONFIGS = {
    1: [6]
}
# CLUSTER_TRAIN_DATA_PATH=TWO_PANEL_TRAIN_CSV
# CLUSTER_TEST_DATA_PATH=TWO_PANEL_TEST_CSV
# 评估模型架构配置
# EVALUATE_LAYER_CONFIGS_TL = [
#     [(1, 6), (10, 0)],
#     [(1, 6), (10, 1)],
#     [(1, 6), (10, 2)],
#     [(1, 6), (10, 3)],
#     [(1, 6), (10, 4)],
#     [(1, 6), (10, 5)],
#     [(1, 6), (10, 6)],
#     [(1, 6), (10, 7)],
#     [(1, 6), (10, 8)],
#     [(1, 6), (10, 9)],
#     [(1, 6), (10, 10)],
#     [(1, 6), (10, 11)],
#     [(1, 6), (10, 12)],
#     [(1, 6), (10, 13)],
#     [(1, 6), (10, 14)],
#     [(1, 6), (10, 15)],
#     [(1, 6), (10, 16)],
#     [(1, 6), (10, 17)],
#     [(1, 6), (10, 18)],
#     [(1, 6), (10, 19)],
#     [(1, 6), (10, 20)],
#     [(1, 6), (10, 21)],
#     [(1, 6), (10, 22)],
#     [(1, 6), (10, 23)],
#     [(1, 6), (10, 24)],
#     [(1, 6), (10, 25)],
#     [(1, 6), (10, 26)],
#     [(1, 6), (10, 27)],
#     [(1, 6), (10, 28)],
#     [(1, 6), (10, 29)],
#     [(1, 6), (10, 30)],
#     [(1, 6), (10, 31)],
#     [(1, 6), (10, 32)],
#     [(1, 6), (10, 33)],
#     [(1, 6), (10, 34)],
#     [(1, 6), (10, 35)],
#     [(1, 6), (10, 36)],
#     [(1, 6), (10, 37)],
#     [(1, 6), (10, 38)],
#     [(1, 6), (10, 39)],
#     [(1, 6), (10, 40)],
#     [(1, 6), (10, 41)],
#     [(1, 6), (10, 42)],
#     [(1, 6), (10, 43)],
#     [(1, 6), (10, 44)],
#     [(1, 6), (10, 45)],
#     [(1, 6), (10, 46)],
#     [(1, 6), (10, 47)],
#     [(1, 6), (10, 48)],
#     [(1, 6), (10, 49)],
#     [(1, 6), (10, 50)],
#     [(1, 6), (10, 51)],
#     [(1, 6), (10, 52)],
#     [(1, 6), (10, 53)],
#     [(1, 6), (10, 54)],
#     [(1, 6), (10, 55)],
#     [(1, 6), (10, 56)],
#     [(1, 6), (10, 57)],
#     [(1, 6), (10, 58)],
#     [(1, 6), (10, 59)],
#     [(1, 6), (10, 60)],
#     [(1, 6), (10, 61)],
#     [(1, 6), (10, 62)],
#     [(1, 6), (10, 63)],
#     [(1, 6), (10, 64)],
#     [(1, 6), (10, 65)],
#     [(1, 6), (10, 66)],
#     [(1, 6), (10, 67)],
#     [(1, 6), (10, 68)],
#     [(1, 6), (10, 69)],
#     [(1, 6), (10, 70)],
#     [(1, 6), (10, 71)],
#     [(1, 6), (10, 72)],
#     [(1, 6), (10, 73)],
#     [(1, 6), (10, 74)],
#     [(1, 6), (10, 75)],
#     [(1, 6), (10, 76)],
#     [(1, 6), (10, 77)],
#     [(1, 6), (10, 78)],
#     [(1, 6), (10, 79)],
#     [(1, 6), (10, 80)],
#     [(1, 6), (10, 81)],
#     [(1, 6), (10, 82)],
#     [(1, 6), (10, 83)],
#     [(1, 6), (10, 84)],
#     [(1, 6), (10, 85)],
#     [(1, 6), (10, 86)],
#     [(1, 6), (10, 87)],
#     [(1, 6), (10, 88)],
#     [(1, 6), (10, 89)],
#     [(1, 6), (10, 90)],
#     [(1, 6), (10, 91)],
#     [(1, 6), (10, 92)],
#     [(1, 6), (10, 93)],
#     [(1, 6), (10, 94)],
#     [(1, 6), (10, 95)],
#     [(1, 6), (10, 96)],
#     [(1, 6), (10, 97)],
#     [(1, 6), (10, 98)],
#     [(1, 6), (10, 99)],
#     [(1, 6), (10, 100)],
#     [(1, 6), (10, 101)],
#     [(1, 6), (10, 102)],
#     [(1, 6), (10, 103)],
#     [(1, 6), (10, 104)],
#     [(1, 6), (10, 105)],
#     [(1, 6), (10, 106)],
#     [(1, 6), (10, 107)],
#     [(1, 6), (10, 108)],
#     [(1, 6), (10, 109)],
#     [(1, 6), (10, 110)],
#     [(1, 6), (10, 111)],
#     [(1, 6), (10, 112)],
#     [(1, 6), (10, 113)],
#     [(1, 6), (10, 114)],
#     [(1, 6), (10, 115)],
#     [(1, 6), (10, 116)],
#     [(1, 6), (10, 117)],
#     [(1, 6), (10, 118)],
#     [(1, 6), (10, 119)],
#     [(1, 6), (10, 120)],
#     [(1, 6), (10, 121)],
#     [(1, 6), (10, 122)],
#     [(1, 6), (10, 123)],
#     [(1, 6), (10, 124)],
#     [(1, 6), (10, 125)],
#     [(1, 6), (10, 126)],
#     [(1, 6), (10, 127)],
#     [(1, 6), (10, 128)],
#     [(1, 6), (10, 129)],
#     [(1, 6), (10, 130)],
#     [(1, 6), (10, 131)],
#     [(1, 6), (10, 132)],
#     [(1, 6), (10, 133)],
#     [(1, 6), (10, 134)],
#     [(1, 6), (10, 135)],
#     [(1, 6), (10, 136)],
#     [(1, 6), (10, 137)],
#     [(1, 6), (10, 138)],
#     [(1, 6), (10, 139)],
#     [(1, 6), (10, 140)],
#     [(1, 6), (10, 141)],
#     [(1, 6), (10, 142)],
#     [(1, 6), (10, 143)],
#     [(1, 6), (10, 144)],
#     [(1, 6), (10, 145)],
#     [(1, 6), (10, 146)],
#     [(1, 6), (10, 147)],
#     [(1, 6), (10, 148)],
#     [(1, 6), (10, 149)],
#     [(1, 6), (10, 150)],
#     [(1, 6), (10, 151)],
#     [(1, 6), (10, 152)],
#     [(1, 6), (10, 153)],
#     [(1, 6), (10, 154)],
#     [(1, 6), (10, 155)],
#     [(1, 6), (10, 156)],
#     [(1, 6), (10, 157)],
#     [(1, 6), (10, 158)],
#     [(1, 6), (10, 159)],
#     [(1, 6), (10, 160)],
#     [(1, 6), (10, 161)],
#     [(1, 6), (10, 162)],
#     [(1, 6), (10, 163)],
#     [(1, 6), (10, 164)],
#     [(1, 6), (10, 165)],
#     [(1, 6), (10, 166)],
#     [(1, 6), (10, 167)],
#     [(1, 6), (10, 168)],
#     [(1, 6), (10, 169)],
#     [(1, 6), (10, 170)],
#     [(1, 6), (10, 171)],
#     [(1, 6), (10, 172)],
#     [(1, 6), (10, 173)],
#     [(1, 6), (10, 174)],
#     [(1, 6), (10, 175)],
#     [(1, 6), (10, 176)],
#     [(1, 6), (10, 177)],
#     [(1, 6), (10, 178)],
#     [(1, 6), (10, 179)],
#     [(1, 6), (10, 180)],
#     [(1, 6), (10, 181)],
#     [(1, 6), (10, 182)],
#     [(1, 6), (10, 183)],
#     [(1, 6), (10, 184)],
#     [(1, 6), (10, 185)],
#     [(1, 6), (10, 186)],
#     [(1, 6), (10, 187)],
#     [(1, 6), (10, 188)],
#     [(1, 6), (10, 189)],
#     [(1, 6), (10, 190)],
#     [(1, 6), (10, 191)],
#     [(1, 6), (10, 192)],
#     [(1, 6), (10, 193)],
#     [(1, 6), (10, 194)],
#     [(1, 6), (10, 195)],
#     [(1, 6), (10, 196)],
#     [(1, 6), (10, 197)],
#     [(1, 6), (10, 198)],
#     [(1, 6), (10, 199)],
# ]
EVALUATE_LAYER_CONFIGS_TL = [
    [(1, 6), (10, 1)],
    ]
# 模型评估样本数量配置
EVALUATE_CLUSTER_TRAIN_SAMPLES = [10]
EVALUATE_CLUSTER_SPLITS = 10
CLUSTER_INDICES = 200

"""
=================================================迁移模型相关路径配置=================================================
"""

# 迁移学习超参数配置
TRANSFER_TRAIN_CONFIG = {
    'patience': DEFAULT_HPARAMS['patience'],
    'lambda_init': DEFAULT_HPARAMS['lambda_init'],
}
# 迁移学习架构配置
# TRANSFER_MODEL_CONFIGS = {
#     "1_4": None,
#     "1_5": None,
#     "1_6": None,
#     "1_10": None,
#     "1_15": None,
#     "2_2": 1,
#     "2_3": 1,
#     "2_4": 1,
#     "2_5": 1,
#     "2_6": 1,
#     "2_10": 1,
#     "2_15": 1,
#     "3_2": 2,
#     "3_3": 2,
#     "3_4": 2,
#     "3_5": 2,
#     "3_6": 2,
#     "3_10": 2,
#     "3_15": 2,
#     "4_2": 3,
#     "4_3": 3,
#     "4_4": 3,
#     "4_5": 3,
#     "4_6": 3,
#     "4_10": 3,
#     "4_15": 3
# }
TRANSFER_MODEL_CONFIGS = {
      "1_4": None,
      "1_5": None,
      "1_6": None,
      "1_10": None,
      "1_15": None,
      "2_2": 1,
      "2_3": 1,
      "2_4": 1,
      "2_5": 1,
      "2_6": 1,
      "2_10": 1,
      "2_15": 1,
}
TL_MULTI_SEED_OUTPUT_DIR = TRANSFER_RESULTS_DIR / "Seed_evaluation"
"""
=================================================外推模型相关路径配置=================================================
"""

FULL_TRAINING_MODEL = "1_6"
ENABLE_DUAL_EXTRAPOLATION = True
"""
===================================================相关配置辅助函数===================================================
"""
# -------------------------- 路径获取函数 --------------------------
def get_alpha_model_path(layer: int, neurons: int) -> Path:
    return NN_RESULTS_DIR / f"alpha_model_{layer}_{neurons}" / "decay_model.pth"
def get_cluster_model_path(layer: int, neurons: int) -> Path:
    return NN_RESULTS_DIR / f"cluster_model_{layer}_{neurons}" / "decay_model.pth"
def get_combined_model_path(layer: int, neurons: int) -> Path:
    return NN_RESULTS_DIR / f"Combined_model_{layer}_{neurons}" / "decay_model.pth"
def get_model_path(model_name: str) -> Path:
    layer, neurons = map(int, model_name.split('_'))
    return get_alpha_model_path(layer, neurons)

# -------------------------- 训练配置函数 --------------------------
def get_alpha_training_config(layer: int, neurons: int) -> dict:
    return {
        'hidden_layers': layer,
        'hidden_neurons': neurons,
        **DEFAULT_HPARAMS
    }
def get_cluster_training_config(layer: int, neurons: int) -> dict:
    config = {
        'hidden_layers': layer,
        'hidden_neurons': neurons,
        'lr': 5e-5,
        'weight_decay': 5e-4,
        **{k: v for k, v in DEFAULT_HPARAMS.items() if k not in ['lr', 'weight_decay']}
    }
    return config
def get_combined_training_config(layer: int, neurons: int) -> dict:
    config = {
        'hidden_layers': layer,
        'hidden_neurons': neurons,
        'lr': 5e-5,
        'weight_decay': 5e-4,
        **{k: v for k, v in DEFAULT_HPARAMS.items() if k not in ['lr', 'weight_decay']}
    }
    return config

# -------------------------- 输出目录函数 --------------------------
def get_alpha_output_dir(layer: int, neurons: int) -> Path:
    return NN_RESULTS_DIR / f"alpha_model_{layer}_{neurons}"
def get_cluster_output_dir(layer: int, neurons: int) -> Path:
    return NN_RESULTS_DIR / f"cluster_model_{layer}_{neurons}"
def get_combined_output_dir(layer: int, neurons: int) -> Path:
    return NN_RESULTS_DIR / f"combined_model_{layer}_{neurons}"


"""
================================================六子图可视化配置================================================
"""
# 增加可视化子目录
SIX_PANEL_DIR = VISUALIZATION_DIR / "six_panel_comparison"
# K-fold RMS文件路径
FOLD_RMS_ALPHA = NN_RESULTS_DIR / "alpha_model_1_6" / "fold_rms_summary.csv"
FOLD_RMS_CLUSTER = TRANSFER_RESULTS_DIR / "model_1_6_k10_row32" / "K-fold_results_cluster_AdamW.csv"
FOLD_RMS_COMBINED = NN_RESULTS_DIR / "combined_model_1_6_train10" / "fold_rms_summary.csv"
# 预测结果CSV文件路径
ALPHA_CSV_PATH = NN_RESULTS_DIR / "evaluate_model_1_6_alpha_only" / "cluster_extrapolation_results.csv"
CLUSTER_ONLY_EXTRAPOLATION_CSV = TRANSFER_RESULTS_DIR / "model_1_6_k10_row32" / "extrapolation_results_cluster_logscale_AdamW.csv"
COMBINED_EXTRAPOLATION_CSV = TRANSFER_RESULTS_DIR / "model_1_6_k10_row32" / "extrapolation_results_combined_logscale_AdamW.csv"
EXPLORATION_PATH = TRANSFER_RESULTS_DIR/ "exploration.csv"

# 六子图输出路径
SIX_PANEL_OUTPUT_PATH = SIX_PANEL_DIR / "six_panel_comparison.png"
SIX_PANEL_RESIDUAL_OUTPUT_PATH = SIX_PANEL_DIR / "six_panel_comparison_residual.png"


"""
================================================种子评估可视化配置================================================
"""
# =============================================== 种子对比可视化目录 ===============================================
SEED_COMPARISON_DIR = VISUALIZATION_DIR / "Seed_evaluation"

# =============================================== 输入文件路径 ===============================================
METRICS_ALL_SEEDS_PATH = MULTI_SEED_OUTPUT_DIR / "metrics_all_seeds.csv"
STATISTICS_SUMMARY_PATH = MULTI_SEED_OUTPUT_DIR / "statistics_summary.csv"
SEED_RESULTS_CLUSTER_PATH = MULTI_SEED_OUTPUT_DIR / "results_cluster.csv"
SEED_RATIO40_RESULTS_CLUSTER_TL_PATH = TL_MULTI_SEED_OUTPUT_DIR / "transfer_seeds_ratio40.csv"
SEED_MERGE_RESULTS_CLUSTER_TL_PATH = TL_MULTI_SEED_OUTPUT_DIR / "merged_predictions.csv"
SEED_RESULTS_ALPHA_PATH = MULTI_SEED_OUTPUT_DIR / "results_alpha.csv"


TWO_PANEL_MODEL_CONFIG = {
    'layer': 1,
    'neurons': 6,
    'train_samples': 10,
    'sort_row': 32
}
TWO_PANEL_TRAIN_CSV = TRANSFER_RESULTS_DIR / f"model_{TWO_PANEL_MODEL_CONFIG['layer']}_{TWO_PANEL_MODEL_CONFIG['neurons']}" / f"train_k{TWO_PANEL_MODEL_CONFIG['train_samples']}_row{TWO_PANEL_MODEL_CONFIG['sort_row']}.csv"
TWO_PANEL_TEST_CSV = TRANSFER_RESULTS_DIR / f"model_{TWO_PANEL_MODEL_CONFIG['layer']}_{TWO_PANEL_MODEL_CONFIG['neurons']}" / f"test_k{TWO_PANEL_MODEL_CONFIG['train_samples']}_row{TWO_PANEL_MODEL_CONFIG['sort_row']}.csv"
TWO_PANEL_CLUSTER_DATA = CLUSTER_DATA_PATH
CLUSTER_TRAIN_DATA_PATH=TWO_PANEL_TRAIN_CSV
CLUSTER_TEST_DATA_PATH=TWO_PANEL_TEST_CSV



"""
================================================核素散点图可视化配置（最终窄幅版）================================================
"""
PLOT_CONFIG = {
    'figsize': (9, 7),          # 宽度适配双行标签
    'dpi': 300,
    'font_family': 'Times New Roman',
    'font_size': {
        'title': 16,
        'subtitle': 14,
        'label': 15,
        'tick': 11,       # 母核/普通刻度字体大小（原配置）
        'tick_child': 9,
        'legend': 11,
        'xlabel': 13
    },
    'colors': {
        'experimental': '#1f77b4',      # 蓝色 (实验值)
        'predicted_uncertain': '#d62728',  # 红色 (DNN)
        'udl_prediction': '#2ca02c',    # 绿色 (UDL)
        'arrow': '#000000'
    },
    'sizes': {
        'experimental': 95,
        'predicted': 115,
        'udl': 88,
        'arrow_length': 3,            # 固定箭头长度 (logT空间)
        'arrow_linewidth': 1.5
    },
    'grid_alpha': 0.65,
    'line_width': {
        'axis': 1.7,
        'grid': 0.85,
        'scatter_edge': 1.3
    }
}
# 调试配置
SHOW_ONLY_UNCERTAIN = True
ISOTOPE_MODEL_NAME = "1_6"
ISOTOPE_K10_CSV = TRANSFER_RESULTS_DIR / "model_1_6" / "extrapolation_k10_best" / "cluster_prediction_summary.csv"
ISOTOPE_FULL_CSV = TRANSFER_RESULTS_DIR / "model_1_6" / "full_training" / "extrapolation_full" / "cluster_prediction_summary.csv"
ISOTOPE_UDL_CSV = ROOT_DIR / "results" / "03_visualization" / "isotope_scatter_vertical" / "udl_extrapolation.csv"

# 输出路径
ISOTOPE_SCATTER_DIR = ROOT_DIR / "results" / "03_visualization" / "isotope_scatter_vertical"
ISOTOPE_OUTPUT_PATH = ISOTOPE_SCATTER_DIR / "scatter_simple.png"



"""
================================================迁移学习RMS对比可视化配置（1_6 vs 2_6）================================================
"""
TL_RMS_SUMMARY_1_6 = TRANSFER_RESULTS_DIR / "model_1_6" / "transfer_rms_summary.csv"
TL_RMS_SUMMARY_2_6 = TRANSFER_RESULTS_DIR / "model_2_6" / "transfer_rms_summary.csv"
TL_SHARED_SUBSET_VIS_CONFIG = {
    "base_model": (1, 6),          # 基准模型（仅用于参考，实际各自独立选子集）
    "best_n": 50,                  # 每个k值下选择的最优子集数量
    "compare_models": [            # 对比模型列表
        (1, 6),                    # 1层6神经元（full fine tuning）
        (2, 6)                     # 2层6神经元（frozen fine tuning）
    ],
    "model_labels": {              # 模型标签（小写+训练策略）
        (1, 6): "1 hidden layer, 6 neurons (full fine tuning)",
        (2, 6): "2 hidden layers, 6 neurons (frozen fine tuning)"
    },
    "model_colors": {              # 模型颜色
        (1, 6): '#1f77b4',         # 蓝色
        (2, 6): '#ff7f0e'          # 橙色
    },
    "output_filename": "transferlearning_rms_comparison.png",  # 注意：无空格
    "title": "Transferlearning RMS comparison",                # 严格无空格
    "show_uncertainty": True       # 所有模型均显示不确定度
}

# 输出目录
TL_SHARED_SUBSET_DIR = VISUALIZATION_DIR / "shared_subset_comparison"
TL_SHARED_SUBSET_OUTPUT_PATH = TL_SHARED_SUBSET_DIR / TL_SHARED_SUBSET_VIS_CONFIG["output_filename"]

"""
================================================ 400子集统一模型评估配置 =================================================
"""
TL_400_SUBSETS_EVAL_CONFIG = {
    "base_model": (1, 6),
    "best_n": 100,
    "feature_cols": ['Z', 'A', 'Q_MeV', 'Z_k', 'A_k'],
    "hidden_layers": 1,      # 与1_6模型一致
    "hidden_neurons": 6,     # 与1_6模型一致
    "train_epochs": 200,     # 足够收敛的最小epochs
    "output_filename": "1_6_400_subsets_independent_evaluation.csv",
    "random_seed_base": 42
}

# 评估输出目录
TL_400_SUBSETS_EVAL_DIR = VISUALIZATION_DIR / "400_subsets_evaluation"
TL_400_SUBSETS_EVAL_OUTPUT_PATH = TL_400_SUBSETS_EVAL_DIR / TL_400_SUBSETS_EVAL_CONFIG["output_filename"]
