import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline import load_data, preprocess_data, feature_engineering

DATA_FILE = "Vehicle Insurance Fraud Detection.csv"

class TestFraudPipeline(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(DATA_FILE), "缺少csv数据文件，跳过本测试")
    def test_load_data_not_empty(self):
        df = load_data(DATA_FILE)
        self.assertGreater(len(df), 0)

    @unittest.skipUnless(os.path.exists(DATA_FILE), "缺少csv数据文件，跳过本测试")
    def test_preprocess_valid(self):
        df_raw = load_data(DATA_FILE)
        df_clean = preprocess_data(df_raw)
        self.assertGreater(len(df_clean), 0)

    @unittest.skipUnless(os.path.exists(DATA_FILE), "缺少csv数据文件，跳过本测试")
    def test_feature_engineering_shape(self):
        df_raw = load_data(DATA_FILE)
        df_clean = preprocess_data(df_raw)
        df_feat = feature_engineering(df_clean)
        self.assertGreater(df_feat.shape[1], df_clean.shape[1])


if __name__ == "__main__":
    unittest.main()
