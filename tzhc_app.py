from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from tzhc_engine import ensure_dirs, load_config, load_prices_csv, run_backtest, save_config, save_result_bundle, write_sample_prices
from tzhc_audit import append_backtest_audit, verify_audit_chain
from tzhc_market import update_watchlist_prices
from tzhc_watchlist import import_legacy_watchlist, load_watchlist


def resolve_root() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir.parent / "config").exists():
            return exe_dir.parent
        return exe_dir
    return Path(__file__).resolve().parent


ROOT = resolve_root()
CONFIG_PATH = ROOT / "config" / "settings.json"
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SAMPLE_CSV = DATA_DIR / "sample_prices.csv"
WATCHLIST_PATH = DATA_DIR / "watchlist.csv"
LEGACY_ROOT = Path(r"F:\xgcx")
MARKET_CSV = DATA_DIR / "a_share_prices.csv"
MARKET_REPORT = DATA_DIR / "market_update_report.json"
AUDIT_LOG = ROOT / "logs" / "backtest_audit.jsonl"


FIELD_DEFS = [
    ("initial_capital", "初始资金", "100000"),
    ("soft_drawdown_pct", "软回撤%（停开新仓）", "5"),
    ("hard_drawdown_pct", "硬回撤%（清仓停机）", "8"),
    ("max_positions", "最多持仓数", "5"),
    ("max_position_pct", "单票仓位%", "18"),
    ("stop_loss_pct", "止损%", "6"),
    ("take_profit_pct", "移动止盈启动%", "14"),
    ("min_volume_ratio", "双高最小量比", "1.8"),
    ("ma_fast", "入场均线", "5"),
    ("ma_trend", "趋势均线", "20"),
    ("ma_long", "长期均线", "60"),
    ("trailing_stop_pct", "高点回撤卖出%", "5"),
    ("lot_size", "每手股数", "100"),
    ("price_limit_pct", "涨跌停幅度%", "10"),
]


class DrawdownApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        ensure_dirs(ROOT)
        self.config_data = load_config(CONFIG_PATH)
        self.selected_csv = tk.StringVar(value=str(MARKET_CSV if MARKET_CSV.exists() else SAMPLE_CSV))
        self.watch_search = tk.StringVar()
        self.vars = {}
        self.last_result = None
        self.watchlist_rows = []
        self.title("量化回撤控制系统 5%-8% 模拟版")
        self.geometry("1280x800")
        self.minsize(1050, 700)
        self._build_ui()
        self._load_watchlist_ui()
        self._log("系统已启动。当前版本只做模拟回测，不接实盘、不保证收益。")
        self._log("回撤规则：5% 暂停开新仓，8% 模拟清仓并停止后续交易。")

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=8)
        style.configure("Accent.TButton", padding=8)

        container = ttk.Frame(self, padding=14)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="量化回撤控制系统", font=("Microsoft YaHei UI", 22, "bold")).pack(side="left")
        ttk.Label(header, text="5%-8% 风控模拟版", foreground="#0a7a5c", font=("Microsoft YaHei UI", 12, "bold")).pack(side="right")

        main = ttk.PanedWindow(container, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=12)
        right = ttk.Frame(main, padding=12)
        main.add(left, weight=1)
        main.add(right, weight=3)

        ttk.Label(left, text="参数设置", font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        for key, label, fallback in FIELD_DEFS:
            row = ttk.Frame(left)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=18).pack(side="left")
            var = tk.StringVar(value=str(self.config_data.get(key, fallback)))
            self.vars[key] = var
            ttk.Entry(row, textvariable=var, width=14).pack(side="right", fill="x", expand=True)

        ttk.Separator(left).pack(fill="x", pady=12)
        ttk.Label(left, text="数据文件 CSV", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        ttk.Entry(left, textvariable=self.selected_csv).pack(fill="x", pady=6)

        file_actions = ttk.Frame(left)
        file_actions.pack(fill="x", pady=3)
        ttk.Button(file_actions, text="生成样例CSV", command=self.generate_sample).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(file_actions, text="选择CSV导入", command=self.choose_csv).pack(side="left", fill="x", expand=True)
        ttk.Button(left, text="保存配置", command=self.save_current_config).pack(fill="x", pady=4)
        self.run_button = ttk.Button(left, text="运行回测", command=self.run_backtest_ui)
        self.run_button.pack(fill="x", pady=8)
        ttk.Button(left, text="打开报告文件夹", command=lambda: self.open_path(REPORT_DIR)).pack(fill="x", pady=4)

        notes = (
            "CSV格式：date,symbol,open,high,low,close,volume\n"
            "先点“生成样例CSV”，再点“运行回测”即可演示。\n\n"
            "风险提示：这里只是历史数据模拟，不构成投资建议。"
        )
        ttk.Label(left, text=notes, foreground="#555", justify="left", wraplength=330).pack(anchor="w", pady=14)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(right, textvariable=self.status_var, font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)

        summary_tab = ttk.Frame(self.tabs, padding=8)
        trades_tab = ttk.Frame(self.tabs, padding=6)
        signals_tab = ttk.Frame(self.tabs, padding=6)
        curve_tab = ttk.Frame(self.tabs, padding=6)
        watchlist_tab = ttk.Frame(self.tabs, padding=6)
        self.tabs.add(summary_tab, text="摘要与日志")
        self.tabs.add(trades_tab, text="交易记录")
        self.tabs.add(signals_tab, text="风控信号")
        self.tabs.add(curve_tab, text="权益/回撤曲线")
        self.tabs.add(watchlist_tab, text="自选股")

        self.log_text = tk.Text(summary_tab, wrap="word", font=("Microsoft YaHei UI", 10), height=28)
        self.log_text.pack(fill="both", expand=True)
        self.trade_tree = ttk.Treeview(trades_tab, columns=("symbol", "entry", "exit", "shares", "pnl", "fees", "reason"), show="headings")
        for key, label, width in (("symbol", "代码", 90), ("entry", "买入日", 95), ("exit", "卖出日", 95), ("shares", "股数", 70), ("pnl", "净盈亏", 85), ("fees", "费用", 70), ("reason", "退出原因", 260)):
            self.trade_tree.heading(key, text=label)
            self.trade_tree.column(key, width=width, anchor="center" if key != "reason" else "w")
        self.trade_tree.pack(fill="both", expand=True)
        self.signal_tree = ttk.Treeview(signals_tab, columns=("date", "symbol", "action", "dd", "reason"), show="headings")
        for key, label, width in (("date", "日期", 100), ("symbol", "代码", 90), ("action", "动作", 110), ("dd", "回撤%", 70), ("reason", "原因", 320)):
            self.signal_tree.heading(key, text=label)
            self.signal_tree.column(key, width=width, anchor="center" if key != "reason" else "w")
        self.signal_tree.pack(fill="both", expand=True)
        self.curve_canvas = tk.Canvas(curve_tab, background="#ffffff", highlightthickness=0)
        self.curve_canvas.pack(fill="both", expand=True)
        self.curve_canvas.bind("<Configure>", lambda _event: self._draw_curve())

        watch_tools = ttk.Frame(watchlist_tab)
        watch_tools.pack(fill="x", pady=(0, 6))
        ttk.Label(watch_tools, text="搜索代码/名称/标签：").pack(side="left")
        watch_entry = ttk.Entry(watch_tools, textvariable=self.watch_search, width=28)
        watch_entry.pack(side="left", padx=(0, 8))
        watch_entry.bind("<KeyRelease>", lambda _event: self._render_watchlist())
        self.watch_count_var = tk.StringVar(value="自选股：0")
        ttk.Label(watch_tools, textvariable=self.watch_count_var).pack(side="right")
        watch_actions = ttk.Frame(watchlist_tab)
        watch_actions.pack(fill="x", pady=(0, 6))
        ttk.Button(watch_actions, text="导入旧系统自选股", command=self.import_old_watchlist).pack(side="left", padx=(0, 5))
        self.market_update_button = ttk.Button(watch_actions, text="更新真实行情", command=self.update_market_data)
        self.market_update_button.pack(side="left", padx=(0, 5))
        ttk.Button(watch_actions, text="使用自选股行情", command=self.use_watchlist_market_data).pack(side="left", padx=(0, 5))
        ttk.Button(watch_actions, text="校验回测审计链", command=self.verify_audit_ui).pack(side="left")
        self.watch_tree = ttk.Treeview(watchlist_tab, columns=("code", "name", "source", "tags"), show="headings")
        for key, label, width in (("code", "代码", 90), ("name", "名称", 110), ("source", "来源", 110), ("tags", "标签", 420)):
            self.watch_tree.heading(key, text=label)
            self.watch_tree.column(key, width=width, anchor="center" if key != "tags" else "w")
        watch_scroll = ttk.Scrollbar(watchlist_tab, orient="vertical", command=self.watch_tree.yview)
        self.watch_tree.configure(yscrollcommand=watch_scroll.set)
        watch_scroll.pack(side="right", fill="y")
        self.watch_tree.pack(fill="both", expand=True)

    def _collect_config(self) -> dict:
        config = load_config(CONFIG_PATH)
        for key, _, _ in FIELD_DEFS:
            raw = self.vars[key].get().strip()
            if key in {"max_positions", "ma_fast", "ma_trend", "ma_long", "lot_size"}:
                config[key] = int(float(raw))
            else:
                config[key] = float(raw)
        if not (0 < config["soft_drawdown_pct"] < config["hard_drawdown_pct"]):
            raise ValueError("软回撤必须大于0，且小于硬回撤。建议 5 和 8。")
        if config["hard_drawdown_pct"] > 12:
            raise ValueError("硬回撤过高。当前系统按稳健演示，建议不超过 8。")
        if config["max_positions"] <= 0 or config["lot_size"] <= 0 or config["max_position_pct"] <= 0:
            raise ValueError("持仓数、每手股数和单票仓位必须大于0。")
        if config["max_positions"] * config["max_position_pct"] > 100:
            raise ValueError("最多持仓数 x 单票仓位不能超过100%。")
        return config

    def _log(self, text: str) -> None:
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def save_current_config(self) -> None:
        try:
            config = self._collect_config()
            save_config(CONFIG_PATH, config)
            self._log(f"配置已保存：{CONFIG_PATH}")
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))

    def generate_sample(self) -> None:
        write_sample_prices(SAMPLE_CSV)
        self.selected_csv.set(str(SAMPLE_CSV))
        self._log(f"已生成样例数据：{SAMPLE_CSV}")

    def choose_csv(self) -> None:
        path = filedialog.askopenfilename(title="选择历史行情CSV", filetypes=[("CSV 文件", "*.csv"), ("全部文件", "*.*")])
        if path:
            self.selected_csv.set(path)
            self._log(f"已选择数据：{path}")

    def run_backtest_ui(self) -> None:
        try:
            config = self._collect_config()
            save_config(CONFIG_PATH, config)
            csv_path = Path(self.selected_csv.get())
            if not csv_path.exists():
                write_sample_prices(SAMPLE_CSV)
                csv_path = SAMPLE_CSV
                self.selected_csv.set(str(csv_path))
                self._log("未找到CSV，已自动生成样例数据。")
        except Exception as exc:
            messagebox.showerror("运行失败", str(exc))
            self._log(f"运行失败：{exc}")
            return

        self.run_button.configure(state="disabled", text="正在回测...")
        self.status_var.set("正在读取数据并回测，请稍候...")

        def worker() -> None:
            try:
                price_map = load_prices_csv(csv_path)
                result = run_backtest(price_map, config)
                paths = save_result_bundle(result, REPORT_DIR)
                audit = append_backtest_audit(result, csv_path, paths.values(), AUDIT_LOG)
                paths["audit_id"] = audit["run_id"]
                self.after(0, lambda: self._show_result(result, paths, len(price_map)))
            except Exception as exc:
                self.after(0, lambda message=str(exc): self._show_error(message))

        threading.Thread(target=worker, daemon=True).start()

    def _show_error(self, message: str) -> None:
        self.run_button.configure(state="normal", text="运行回测")
        self.status_var.set("回测失败")
        self._log(f"运行失败：{message}")
        messagebox.showerror("运行失败", message)

    def _show_result(self, result, paths: dict, symbol_count: int) -> None:
        self.last_result = result
        self.run_button.configure(state="normal", text="运行回测")
        self.status_var.set(f"完成：收益 {result.total_return_pct:.2f}%｜最大回撤 {result.max_drawdown_pct:.2f}%｜费用 {result.total_fees:.2f}")
        self._log("-" * 64)
        self._log(f"回测完成：{symbol_count} 个标的；期末权益：{result.final_equity:.2f}")
        self._log(f"总收益：{result.total_return_pct:.2f}%｜年化：{result.annual_return_pct:.2f}%｜最大回撤：{result.max_drawdown_pct:.2f}%")
        self._log(f"交易：{result.trade_count}｜胜率：{result.win_rate_pct:.2f}%｜夏普：{result.sharpe_ratio:.2f}｜盈亏因子：{result.profit_factor:.2f}")
        self._log(f"5%暂停/8%熔断：{'已于 ' + str(result.halt_date) + ' 熔断' if result.halted else '未触发硬熔断'}")
        self._log(f"报告：{paths['report']}")
        self._log(f"审计记录：{paths.get('audit_id', '--')}，数据和结果哈希已写入 {AUDIT_LOG}")
        for tree in (self.trade_tree, self.signal_tree):
            tree.delete(*tree.get_children())
        for trade in result.trades:
            self.trade_tree.insert("", "end", values=(trade.symbol, trade.entry_date, trade.exit_date, trade.shares, f"{trade.pnl:.2f}", f"{trade.fees:.2f}", trade.reason))
        for signal in result.signals:
            self.signal_tree.insert("", "end", values=(signal.date, signal.symbol, signal.action, f"{signal.drawdown_pct:.2f}", signal.reason))
        self._draw_curve()
        messagebox.showinfo("完成", f"回测完成。\n最大回撤：{result.max_drawdown_pct:.2f}%\n报告：{paths['report']}")

    def _draw_curve(self) -> None:
        canvas = self.curve_canvas
        canvas.delete("all")
        if not self.last_result or len(self.last_result.equity_curve) < 2:
            canvas.create_text(20, 20, anchor="nw", text="运行回测后显示权益和回撤曲线", fill="#666666")
            return
        width = max(200, canvas.winfo_width())
        height = max(160, canvas.winfo_height())
        margin = 42
        points = self.last_result.equity_curve
        equities = [item["equity"] for item in points]
        drawdowns = [item["drawdown_pct"] for item in points]
        low, high = min(equities), max(equities)
        span = max(1.0, high - low)
        dd_scale = max(10.0, max(drawdowns))
        x = lambda idx: margin + idx * (width - margin * 2) / (len(points) - 1)
        equity_y = lambda value: margin + (high - value) * (height - margin * 2) / span
        dd_y = lambda value: margin + value * (height - margin * 2) / dd_scale
        canvas.create_line(margin, margin, margin, height - margin, width - margin, height - margin, fill="#999999")
        equity_coords = [coord for idx, value in enumerate(equities) for coord in (x(idx), equity_y(value))]
        dd_coords = [coord for idx, value in enumerate(drawdowns) for coord in (x(idx), dd_y(value))]
        canvas.create_line(*equity_coords, fill="#087f5b", width=2)
        canvas.create_line(*dd_coords, fill="#c92a2a", width=2)
        for threshold, color in ((5, "#e67700"), (8, "#c92a2a")):
            y = dd_y(threshold)
            canvas.create_line(margin, y, width - margin, y, fill=color, dash=(5, 4))
            canvas.create_text(width - margin, y - 8, anchor="e", text=f"回撤 {threshold}%", fill=color)
        canvas.create_text(margin, 14, anchor="w", text=f"权益 {low:.0f} - {high:.0f}", fill="#087f5b")
        canvas.create_text(width - margin, 14, anchor="e", text="绿色：权益　红色：回撤", fill="#444444")

    def import_old_watchlist(self) -> None:
        try:
            self.watchlist_rows = import_legacy_watchlist(LEGACY_ROOT, WATCHLIST_PATH)
            self.watch_search.set("")
            self._render_watchlist()
            self.tabs.select(self.tabs.index("end") - 1)
            self.status_var.set(f"已导入 {len(self.watchlist_rows)} 只自选股")
            self._log(f"旧系统自选股导入完成：{len(self.watchlist_rows)} 只，保存到 {WATCHLIST_PATH}")
            messagebox.showinfo("导入完成", f"已从旧系统合并导入 {len(self.watchlist_rows)} 只自选股。\n重复代码已自动去重。")
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def _load_watchlist_ui(self) -> None:
        self.watchlist_rows = load_watchlist(WATCHLIST_PATH)
        self._render_watchlist()

    def _render_watchlist(self) -> None:
        if not hasattr(self, "watch_tree"):
            return
        query = self.watch_search.get().strip().lower()
        rows = [
            item for item in self.watchlist_rows
            if not query or query in f"{item.code} {item.name} {item.tags} {item.source}".lower()
        ]
        self.watch_tree.delete(*self.watch_tree.get_children())
        for item in rows:
            self.watch_tree.insert("", "end", values=(item.code, item.name, item.source, item.tags))
        self.watch_count_var.set(f"显示 {len(rows)} / 共 {len(self.watchlist_rows)} 只")

    def update_market_data(self) -> None:
        if not self.watchlist_rows:
            messagebox.showerror("无法更新", "自选股为空，请先导入旧系统自选股。")
            return
        self.market_update_button.configure(state="disabled", text="正在更新...")
        self.status_var.set(f"准备更新 {len(self.watchlist_rows)} 只自选股行情...")

        def progress(done: int, total: int, message: str) -> None:
            self.after(0, lambda: self.status_var.set(f"行情更新 {done}/{total}｜{message}"))

        def worker() -> None:
            try:
                result = update_watchlist_prices(self.watchlist_rows, MARKET_CSV, MARKET_REPORT, progress=progress)
                self.after(0, lambda: self._market_update_done(result))
            except Exception as exc:
                self.after(0, lambda message=str(exc): self._market_update_failed(message))

        threading.Thread(target=worker, daemon=True).start()

    def _market_update_done(self, result) -> None:
        self.market_update_button.configure(state="normal", text="更新真实行情")
        self.selected_csv.set(str(MARKET_CSV))
        self.status_var.set(f"行情完成：成功 {result.succeeded}/{result.requested}｜最新 {result.latest_date}")
        self._log(f"真实行情更新完成：成功 {result.succeeded}，失败 {result.failed}，共 {result.rows} 条日线。")
        self._log(f"行情质量报告：{MARKET_REPORT}")
        messagebox.showinfo("行情更新完成", f"成功：{result.succeeded}/{result.requested}\n失败：{result.failed}\n最新交易日：{result.latest_date}\n现在可以运行回测。")

    def _market_update_failed(self, message: str) -> None:
        self.market_update_button.configure(state="normal", text="更新真实行情")
        self.status_var.set("行情更新失败")
        self._log(f"行情更新失败：{message}")
        messagebox.showerror("行情更新失败", message)

    def use_watchlist_market_data(self) -> None:
        if not MARKET_CSV.exists():
            messagebox.showinfo("尚无行情", "请先点击“更新真实行情”。")
            return
        self.selected_csv.set(str(MARKET_CSV))
        self.tabs.select(0)
        self.status_var.set("已选择自选股真实行情，可以点击运行回测")

    def verify_audit_ui(self) -> None:
        try:
            ok, _count, message = verify_audit_chain(AUDIT_LOG)
        except Exception as exc:
            ok, message = False, f"审计链无法读取：{exc}"
        self.status_var.set(message)
        if ok:
            messagebox.showinfo("审计校验通过", message)
        else:
            messagebox.showerror("审计校验失败", message)

    @staticmethod
    def open_path(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        else:
            subprocess.Popen(["xdg-open", str(path)])


def run_selftest() -> None:
    ensure_dirs(ROOT)
    config = load_config(CONFIG_PATH)
    write_sample_prices(SAMPLE_CSV)
    price_map = load_prices_csv(SAMPLE_CSV)
    result = run_backtest(price_map, config)
    paths = save_result_bundle(result, REPORT_DIR)
    append_backtest_audit(result, SAMPLE_CSV, paths.values(), AUDIT_LOG)
    print("SELFTEST_OK")
    print(f"sample={SAMPLE_CSV}")
    print(f"report={paths['report']}")
    print(f"final_equity={result.final_equity}")
    print(f"max_drawdown={result.max_drawdown_pct}%")


def run_year_backtest(year: int) -> None:
    ensure_dirs(ROOT)
    if not MARKET_CSV.exists():
        raise FileNotFoundError("请先更新自选股真实行情。")
    full_map = load_prices_csv(MARKET_CSV)
    prefix = f"{year:04d}-"
    price_map = {symbol: [row for row in rows if row.date.startswith(prefix)] for symbol, rows in full_map.items()}
    price_map = {symbol: rows for symbol, rows in price_map.items() if rows}
    if not price_map:
        raise ValueError(f"真实行情中没有 {year} 年数据。")
    year_csv = DATA_DIR / f"a_share_prices_{year}.csv"
    with year_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
        for symbol in sorted(price_map):
            for row in price_map[symbol]:
                writer.writerow([row.date, row.symbol, row.open, row.high, row.low, row.close, row.volume])
    result = run_backtest(price_map, load_config(CONFIG_PATH))
    paths = save_result_bundle(result, REPORT_DIR, run_label=str(year))
    audit = append_backtest_audit(result, year_csv, paths.values(), AUDIT_LOG)
    print(f"YEAR_BACKTEST={year}")
    print(f"RETURN={result.total_return_pct}% MAX_DD={result.max_drawdown_pct}% HALTED={result.halted}")
    print(f"REPORT={paths['report']}")
    print(f"AUDIT={audit['run_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="量化回撤控制系统 5%-8% 模拟版")
    parser.add_argument("--selftest", action="store_true", help="生成样例数据并跑一次回测")
    parser.add_argument("--sample", action="store_true", help="只生成样例CSV")
    parser.add_argument("--import-watchlist", nargs="?", const=str(LEGACY_ROOT), help="导入旧系统自选股目录")
    parser.add_argument("--update-market", action="store_true", help="更新全部自选股真实日线")
    parser.add_argument("--verify-audit", action="store_true", help="校验回测审计链")
    parser.add_argument("--backtest-year", type=int, help="按指定年份重新起算组合回测")
    args = parser.parse_args()

    ensure_dirs(ROOT)
    if args.sample:
        write_sample_prices(SAMPLE_CSV)
        print(SAMPLE_CSV)
        return
    if args.selftest:
        run_selftest()
        return
    if args.import_watchlist:
        items = import_legacy_watchlist(Path(args.import_watchlist), WATCHLIST_PATH)
        print(f"WATCHLIST_IMPORTED={len(items)}")
        print(WATCHLIST_PATH)
        return
    if args.update_market:
        items = load_watchlist(WATCHLIST_PATH)
        result = update_watchlist_prices(items, MARKET_CSV, MARKET_REPORT)
        print(f"MARKET_UPDATED={result.succeeded}/{result.requested}")
        print(MARKET_CSV)
        return
    if args.verify_audit:
        ok, count, message = verify_audit_chain(AUDIT_LOG)
        print(message)
        raise SystemExit(0 if ok else 2)
    if args.backtest_year:
        run_year_backtest(args.backtest_year)
        return
    app = DrawdownApp()
    app.mainloop()


if __name__ == "__main__":
    main()
