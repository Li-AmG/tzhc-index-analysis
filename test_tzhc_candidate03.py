import tempfile
import unittest
from pathlib import Path

from tzhc_candidate03 import (
    add_candidate03_holding,
    load_candidate03_holdings,
    remove_candidate03_holding,
)


class Candidate03HoldingTests(unittest.TestCase):
    def test_add_load_remove_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "holdings.csv"
            added = add_candidate03_holding(
                path, "600000", "浦发银行", "2026-07-16", 10.25, 1000, "同花顺手工成交"
            )
            self.assertEqual(len(added), 1)
            loaded = load_candidate03_holdings(path)
            self.assertEqual(loaded.iloc[0]["symbol"], "600000")
            self.assertEqual(int(loaded.iloc[0]["shares"]), 1000)
            removed = remove_candidate03_holding(path, "600000")
            self.assertTrue(removed.empty)

    def test_rejects_duplicate_and_beijing_code(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "holdings.csv"
            add_candidate03_holding(path, "000001", "平安银行", "2026-07-16", 10, 100)
            with self.assertRaises(ValueError):
                add_candidate03_holding(path, "000001", "平安银行", "2026-07-16", 10, 100)
            with self.assertRaises(ValueError):
                add_candidate03_holding(path, "830001", "北交所", "2026-07-16", 10, 100)


if __name__ == "__main__":
    unittest.main()
