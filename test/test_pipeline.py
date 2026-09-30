import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline import load_data, clean_data, feature_engineering

DATA_FILE = "Vehicle Insurance Fraud Detection.csv"

class TestFraudPipeline(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(DATA_FILE), "缺少csv数据文件，跳过本测试")
    def test_load_data_not_empty(self):
        df = load_data(DATA_FILE)
        self.assertGreater(len(df), 0)

    @unittest.skipUnless(os.path.exists(DATA_FILE), "缺少csv数据文件，跳过本测试")
    def test_clean_data_valid(self):
        df_raw = load_data(DATA_FILE)
        df_clean = clean_data(df_raw)
        self.assertGreater(len(df_clean), 0)

    @unittest.skipUnless(os.path.exists(DATA_FILE), "缺少csv数据文件，跳过本测试")
    def test_feature_engineering_shape_match(self):
        df_raw = load_data(DATA_FILE)
        df_clean = clean_data(df_raw)
        X, y, _ = feature_engineering(df_clean, target_col="FraudFound")
        self.assertEqual(len(X), len(y))

if __name__ == "__main__":
    unittest.main()
