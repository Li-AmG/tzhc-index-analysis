from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tzhc_watchlist import add_watch_item, import_legacy_watchlist, load_watchlist


class WatchlistImportTest(unittest.TestCase):
    def test_sel_and_manual_are_merged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "自选股.sel").write_bytes(b"\x07!000001\x07!300001\x07!000001")
            (root / "手工追加自选.txt").write_text(
                "# comment\n000001,平安银行,重点\n600000,浦发银行,观察\n",
                encoding="utf-8",
            )
            output = root / "new" / "watchlist.csv"
            items = import_legacy_watchlist(root, output)
            self.assertEqual(["000001", "300001", "600000"], [item.code for item in items])
            self.assertEqual("平安银行", items[0].name)
            self.assertEqual("同花顺+手工", items[0].source)
            self.assertEqual(3, len(load_watchlist(output)))

    def test_direct_code_add_is_validated_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "watchlist.csv"
            item, added = add_watch_item(output, "301584", "")
            self.assertTrue(added)
            self.assertEqual("301584", item.name)
            self.assertIn("待更新行情", item.tags)
            updated, added_again = add_watch_item(output, "301584", "西测测试")
            self.assertFalse(added_again)
            self.assertEqual("西测测试", updated.name)
            self.assertEqual(1, len(load_watchlist(output)))
            with self.assertRaisesRegex(ValueError, "6位A股代码"):
                add_watch_item(output, "123")


if __name__ == "__main__":
    unittest.main()
