import os
import logging
from pipeline import run_pipeline

def main():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/pipeline.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y‑%m‑%d %H:%M:%S"
    )
    print("===== 车险欺诈检测流水线开始运行 =====")
    results_df, roc_data = run_pipeline("Vehicle Insurance Fraud Detection.csv")
    print("\n流水线执行完成，日志存放在 logs/pipeline.log")
    return results_df, roc_data

if __name__ == "__main__":
    main()
