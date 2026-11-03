from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tzhc_clients import (
    export_tdx_watchlist,
    import_client_watchlist,
    parse_tdx_watchlist,
    parse_ths_watchlist,
)
from tzhc_watchlist import WatchItem, load_watchlist, save_watchlist


class ClientIntegrationTest(unittest.TestCase):
    def test_tdx_and_ths_formats_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tdx = root / "zxg.blk"
            tdx.write_text("1000001\r\n0600000\r\n1999999\r\n", encoding="gb18030")
            self.assertEqual(["000001", "600000"], parse_tdx_watchlist(tdx))
            ths = root / "SelfStockInfo.json"
            ths.write_text(
                json.dumps([{"C": "300001"}, {"C": "600000"}, {"C": "USDCNH"}]),
                encoding="utf-8",
            )
            self.assertEqual(["300001", "600000"], parse_ths_watchlist(ths))

    def test_client_import_merges_and_preserves_existing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system = root / "watchlist.csv"
            save_watchlist([WatchItem("600000", "浦发银行", "重点", "手工")], system)
            source = root / "SelfStockInfo.json"
            source.write_text(json.dumps([{"C": "600000"}, {"C": "300001"}]), encoding="utf-8")
            result = import_client_watchlist("同花顺客户端", source, system)
            rows = load_watchlist(system)
            self.assertEqual(1, result.added_count)
            self.assertEqual("浦发银行", rows[0].name)
            self.assertEqual("重点", rows[0].tags)
            self.assertIn("同花顺客户端", rows[0].source)

    @patch("tzhc_clients.client_process_running", return_value=False)
    def test_tdx_export_merges_and_backs_up(self, _running) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "blocknew" / "zxg.blk"
            target.parent.mkdir(parents=True)
            target.write_text("1999999\r\n1000001\r\n", encoding="gb18030")
            result = export_tdx_watchlist(["600000", "300001"], target, root / "backups", True)
            self.assertEqual(["000001", "600000", "300001"], parse_tdx_watchlist(target))
            self.assertEqual(3, result["after"])
            self.assertEqual(1, len(list((root / "backups").glob("*.blk"))))


if __name__ == "__main__":
    unittest.main()
