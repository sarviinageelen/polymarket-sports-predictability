import csv
import os
import tempfile
import unittest

import fetch_sports as fps


class TestSportsHelpers(unittest.TestCase):
    def test_get_category_known_and_unknown(self):
        self.assertEqual(fps.get_category("nba"), "basketball")
        self.assertEqual(fps.get_category("unknown_code"), "unknown")

    def test_save_to_csv_writes_category(self):
        data = [
            {"sport": "nba", "image": "img1", "resolution": "r1", "ordering": 1},
            {"sport": "mystery", "image": "img2", "resolution": "r2", "ordering": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "sports.csv")
            fps.save_to_csv(data, csv_path)

            with open(csv_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["category"], "basketball")
        self.assertEqual(rows[1]["category"], "unknown")


if __name__ == "__main__":
    unittest.main()
