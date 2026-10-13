import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"Age": 30}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )

    def test_empty_request_validation(self):
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 400)

    def test_invalid_json_validation(self):
        response = self.client.post(
            "/predict",
            data="not valid json",
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
