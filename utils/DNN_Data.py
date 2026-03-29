from utils.utils_Data import *
from utils.utils_Linear import *
import config

"""
数据读取与检验集成调用函数
1、结合config调用数据处理函数   ----   根据项目相对路径自适应输出结构
2、集成调用数据处理函数         ----   简化形参配置及集成结果分析
"""

#===================================================数据加载===================================================
def prepare_data(JSON_FILE_PATH: str | Path = config.JSON_FILE_PATH):

    print("====================== DATA PREPARATION STARTED ======================")
    alpha_df = load_alpha_data(JSON_FILE_PATH)
    print(f"Alpha data extracted: {len(alpha_df)} samples")
    cluster_df = load_cluster_data()
    print(f"Cluster data extracted: {len(cluster_df)} samples")

    ALPHA_DATA_PATH=config.ALPHA_DATA_PATH
    CLUSTER_DATA_PATH=config.CLUSTER_DATA_PATH
    alpha_df.to_csv(ALPHA_DATA_PATH, index=False)
    cluster_df.to_csv(CLUSTER_DATA_PATH, index=False)
    print(f"\nAlpha data saved to: {ALPHA_DATA_PATH}")
    print(f"Cluster data saved to: {CLUSTER_DATA_PATH}")



#===================================================数据检验===================================================
def run_udl_verification_direct():

    print("====================== UDL VERIFICATION ======================")

    try:
        summary_df = main_fit_all(str(config.ALPHA_DATA_PATH), str(config.CLUSTER_DATA_PATH), str(config.DATA_DIR))
        print("UDL verification completed successfully.")
        return summary_df
    except Exception as e:
        print(f"UDL verification failed: {str(e)}")
        return None