import json
import os
import unittest

import joblib
import pandas as pd


class TestSleepDisorderPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("sleep_dataset_processed.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("sleep_disorder_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r", encoding="utf-8") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_model_prediction(self):
        saved = joblib.load("sleep_disorder_model.pkl")
        data = pd.read_csv("sleep_dataset_processed.csv")
        sample = data[saved["features"]].iloc[[0]]

        prediction = saved["model"].predict(sample)[0]
        self.assertIn(str(prediction), saved["labels"])

    def test_multiple_predictions(self):
        saved = joblib.load("sleep_disorder_model.pkl")
        data = pd.read_csv("sleep_dataset_processed.csv")
        sample = data[saved["features"]].iloc[:10]

        predictions = saved["model"].predict(sample)
        for prediction in predictions:
            self.assertIn(str(prediction), saved["labels"])


if __name__ == "__main__":
    unittest.main()
