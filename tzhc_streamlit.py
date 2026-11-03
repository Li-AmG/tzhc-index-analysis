from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from tzhc_clients import (
    client_process_running,
    discover_client_sources,
    export_tdx_watchlist,
    import_client_watchlist,
    parse_tdx_watchlist,
    parse_ths_watchlist,
)
from tzhc_dashboard_signals import calculate_signals, scan_watchlist_signals
from tzhc_engine import Candle, load_prices_csv
from tzhc_market import (
    latest_completed_market_date,
    refresh_watchlist_names,
    update_benchmark_prices,
    update_watchlist_prices,
)
from tzhc_paper import (
    PaperPaths,
    RISK_HARD_LOCK,
    RISK_NORMAL,
    RISK_SOFT_LOCK,
    add_paper_watch_symbol,
    advance_paper_account,
    initialize_paper_account,
    load_paper_account,
    queue_paper_buy_signal,
    remove_paper_watch_symbol,
    unlock_soft_risk,
)
from tzhc_signal_analytics import analyze_signals, save_analytics_bundle
from tzhc_breakout_strategy import analyze_breakout_strategy, run_breakout_portfolios, save_breakout_bundle
from tzhc_candidate03 import (
    add_candidate03_holding,
    load_candidate03_holdings,
    remove_candidate03_holding,
)
from tzhc_ma_strategy import analyze_ma_strategy, run_ma_portfolio_variants, save_ma_strategy_bundle
from tzhc_pullback_strategy import (
    PULLBACK_STRATEGY,
    calculate_pullback_signals,
    is_growth_board,
    rules_from_config as pullback_rules_from_config,
    scan_pullback_signals,
)
from tzhc_lifecycle_strategy import (
    LIFECYCLE_LABEL,
    LIFECYCLE_STRATEGY,
    calculate_lifecycle_signals,
    latest_signal_is_valid as lifecycle_signal_is_valid,
    rules_from_config as lifecycle_rules_from_config,
    scan_lifecycle_signals,
)
from tzhc_limit_assistant import event_forward_returns, scan_limit_assistant
from tzhc_growth_rank import add_rank_to_watchlist, fetch_growth_top_rank
from tzhc_turtle import analyze_turtle_events, save_turtle_bundle
from tzhc_turtle_reversal import analyze_turtle_reversal, save_turtle_reversal_bundle
from tzhc_watchlist import SUPPORTED_STOCK_CODE, add_watch_item, load_watchlist as load_watch_items


ROOT = Path(__file__).resolve().parent
PRICE_PATH = ROOT / "data" / "a_share_prices.csv"
WATCHLIST_PATH = ROOT / "data" / "watchlist.csv"
CONFIG_PATH = ROOT / "config" / "settings.json"
REPORT_DIR = ROOT / "reports"
MARKET_REPORT = ROOT / "data" / "market_update_report.json"
SIGNAL_SCAN_PATH = ROOT / "data" / "watchlist_signal_scan.csv"
LIMIT_ASSISTANT_SCAN_PATH = ROOT / "data" / "limit_assistant_scan.csv"
SIGNAL_ANALYTICS_DIR = REPORT_DIR / "signal_analytics"
BENCHMARK_PATH = ROOT / "data" / "benchmark_prices.csv"
BENCHMARK_REPORT = ROOT / "data" / "benchmark_update_report.json"
TURTLE_ANALYTICS_DIR = REPORT_DIR / "turtle_analytics"
TURTLE_EVENTS_PATH = TURTLE_ANALYTICS_DIR / "events_latest.csv"
TURTLE_REVERSAL_DIR = REPORT_DIR / "turtle_reversal_research"
GROWTH_TOP15_PATH = ROOT / "data" / "growth_board_top15_realtime.csv"
MA_STRATEGY_DIR = REPORT_DIR / "ma_strategy"
BREAKOUT_STRATEGY_DIR = REPORT_DIR / "breakout_strategy"
QUALITY_CANDIDATES_PATH = REPORT_DIR / "quality_trend_candidates.csv"
VALUE_GROWTH_DIR = REPORT_DIR / "value_growth_pullback_backtest"
VALUE_GROWTH_SCRIPT = REPORT_DIR / "value_growth_pullback_backtest.py"
VALUE_GROWTH_REPORT = VALUE_GROWTH_DIR / "value_growth_pullback_report.md"
VALUE_GROWTH_CANDIDATES = VALUE_GROWTH_DIR / "value_growth_pullback_candidates.csv"
LEGACY_RESEARCH_ENABLED = False
PULLBACK_SCAN_PATH = ROOT / "data" / "pullback_ignition_scan.csv"
LIFECYCLE_SCAN_PATH = ROOT / "data" / "ghost_lifecycle_scan.csv"
CANDIDATE03_STATUS_PATH = ROOT / "data" / "candidate03_live_status.json"
CANDIDATE03_BUY_PATH = ROOT / "data" / "candidate03_buy_signals.csv"
CANDIDATE03_LAST_WEEK_PATH = ROOT / "data" / "candidate03_last_week_candidates.csv"
CANDIDATE03_SELL_PATH = ROOT / "data" / "candidate03_sell_signals.csv"
CANDIDATE03_HOLDINGS_PATH = ROOT / "data" / "candidate03_holdings.csv"
CANDIDATE03_REFRESH_PATH = ROOT / "data" / "candidate03_refresh_report.json"
CANDIDATE03_ENGINE = REPORT_DIR / "candidate03_live.py"
CANDIDATE03_RUN_LOCK = ROOT / "data" / "candidate03_live.lock"
IWENCAI_ENGINE = REPORT_DIR / "ths_iwencai_automation.py"
IWENCAI_RESPONSE_PATH = ROOT / "data" / "iwencai_bridge" / "last_response.json"
MARKET_DATA_PYTHON = REPORT_DIR / ".market_data_venv" / "Scripts" / "python.exe"
PAPER_PATHS = PaperPaths.for_root(ROOT)
CLIENT_BACKUP_DIR = ROOT / "data" / "client_backups"
PAPER_RISK_LABELS = {
    RISK_NORMAL: "正常 · 可按规则开仓",
    RISK_SOFT_LOCK: "5%风险锁 · 暂停开仓，等待人工复核",
    RISK_HARD_LOCK: "8%硬熔断 · 禁止开仓，等待清仓或重置账户",
}


st.set_page_config(page_title="量化回撤动态看板", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data(show_spinner=False)
def load_prices(path: str, modified: float) -> pd.DataFrame:
    del modified
    data = pd.read_csv(path, dtype={"symbol": str})
    data["symbol"] = data["symbol"].str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    return data.dropna(subset=["date", *numeric]).sort_values(["symbol", "date"])


@st.cache_data(show_spinner=False)
def load_watchlist(path: str, modified: float) -> pd.DataFrame:
    del modified
    data = pd.read_csv(path, dtype={"code": str}).fillna("")
    data["code"] = data["code"].str.zfill(6)
    return data


@st.cache_data(show_spinner=False)
def load_signal_scan(path: str, modified: float) -> pd.DataFrame:
    del modified
    data = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig").fillna("")
    data["symbol"] = data["symbol"].str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["age_trade_days"] = pd.to_numeric(data["age_trade_days"], errors="coerce").fillna(99).astype(int)
    return data


@st.cache_data(show_spinner=False)
def load_analytics_csv(path: str, modified: float) -> pd.DataFrame:
    del modified
    try:
        return pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run_candidate03(command: str) -> tuple[bool, str]:
    if command not in {"scan", "refresh"}:
        return False, "不支持的候选03命令"
    if not MARKET_DATA_PYTHON.exists() or not CANDIDATE03_ENGINE.exists():
        return False, "候选03行情环境或扫描引擎缺失"
    # Streamlit can have two browser tabs/sessions execute the callback at
    # once.  Keep refresh and scan single-flight so they never race for the
    # DuckDB file lock.
    try:
        CANDIDATE03_RUN_LOCK.mkdir(parents=False)
    except FileExistsError:
        try:
            age = time.time() - CANDIDATE03_RUN_LOCK.stat().st_mtime
        except OSError:
            age = 0
        if age > 3600:
            try:
                # The lock is a directory containing owner.txt, so rmdir()
                # alone can never remove an expired lock.  Remove only the
                # marker file first, then recreate the single-flight lock.
                (CANDIDATE03_RUN_LOCK / "owner.txt").unlink(missing_ok=True)
                CANDIDATE03_RUN_LOCK.rmdir()
                CANDIDATE03_RUN_LOCK.mkdir(parents=False)
            except OSError:
                return False, "候选03仍有一个更新/扫描任务在运行，请稍后再试。"
        else:
            return False, "候选03正在更新或扫描，请等当前任务完成后再试。"
    try:
        (CANDIDATE03_RUN_LOCK / "owner.txt").write_text(
            f"pid={__import__('os').getpid()}\ncommand={command}\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    # Desktop UI automation must inherit the interactive Windows desktop;
    # hiding this child process can place it outside the user's UI session.
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = None
    try:
        process = subprocess.Popen(
            [str(MARKET_DATA_PYTHON), str(CANDIDATE03_ENGINE), command],
            cwd=str(REPORT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )
        stdout, stderr = process.communicate(timeout=1200)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        if process is not None:
            try:
                # Refresh can create helper children while rebuilding DuckDB;
                # terminate this owned process tree so the lock is released.
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=20,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        return False, "候选03更新/扫描超过20分钟，已终止本次任务并释放数据库锁；可先点击‘仅重扫本地数据库’。"
    except OSError as error:
        return False, f"候选03进程启动失败：{error}"
    finally:
        try:
            (CANDIDATE03_RUN_LOCK / "owner.txt").unlink(missing_ok=True)
            CANDIDATE03_RUN_LOCK.rmdir()
        except OSError:
            pass
    output = (stderr or stdout or "").strip()
    if returncode:
        return False, output[-1800:] or f"扫描进程返回 {returncode}"
    if command == "refresh":
        try:
            status_payload = json.loads(
                CANDIDATE03_STATUS_PATH.read_text(encoding="utf-8")
            )
            refresh_payload = status_payload.get("refresh", {})
            if not isinstance(refresh_payload, dict) or not refresh_payload:
                refresh_payload = json.loads(
                    CANDIDATE03_REFRESH_PATH.read_text(encoding="utf-8")
                )
        except (OSError, json.JSONDecodeError):
            return False, "候选03更新进程已结束，但无法读取刷新报告，不能确认行情是否升级。"

        before = str(refresh_payload.get("started_from", "--"))
        after = str(refresh_payload.get("database_date", "--"))
        target = str(
            refresh_payload.get("market", {}).get("latest_requested_date", "")
            if isinstance(refresh_payload.get("market"), dict)
            else ""
        ) or str(refresh_payload.get("cutoff", "--"))
        state = str(refresh_payload.get("status", "stale"))
        advanced = bool(refresh_payload.get("database_advanced"))
        warnings = [
            str(item) for item in refresh_payload.get("warnings", [])
            if str(item).strip()
        ]
        suffix = f" 提示：{warnings[0]}" if warnings else ""
        if advanced:
            return True, f"候选03行情已从 {before} 升级到 {after}；目标交易日 {target}。{suffix}".strip()
        if state == "current":
            return True, f"候选03行情已经是最新交易日 {after}，本次无需新增。{suffix}".strip()
        if state == "local_only":
            return False, f"候选03未能联网，仍使用 {after} 的本地行情。{suffix}".strip()
        return False, (
            f"候选03行情没有升级，数据库仍为 {after}，目标交易日为 {target}。"
            f"{suffix} 本次扫描完成不等于行情更新成功。"
        ).strip()
    return True, "候选03已按本地数据库重新扫描。"


def load_candidate03_status() -> dict:
    if not CANDIDATE03_STATUS_PATH.exists():
        return {}
    try:
        return json.loads(CANDIDATE03_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_iwencai(question: str) -> tuple[bool, str]:
    if not IWENCAI_ENGINE.exists():
        return False, "问财自动化脚本缺失"
    creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    try:
        completed = subprocess.run(
            [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(IWENCAI_ENGINE),
             "--question", question, "--timeout", "90"],
            cwd=str(REPORT_DIR), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120, creationflags=creation_flags, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    output = (completed.stderr or completed.stdout or "").strip()
    if completed.returncode:
        return False, output[-1800:] or "问财自动化失败"
    return True, output[-1000:]


def load_iwencai_response() -> dict:
    if not IWENCAI_RESPONSE_PATH.exists():
        return {}
    try:
        return json.loads(IWENCAI_RESPONSE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def load_benchmarks(path: str, modified: float) -> pd.DataFrame:
    del modified
    data = pd.read_csv(path, dtype={"benchmark_code": str}, encoding="utf-8-sig")
    data["benchmark_code"] = data["benchmark_code"].str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    return data.dropna(subset=["date", *numeric]).sort_values(["benchmark_code", "date"])


def latest_file(pattern: str) -> Path | None:
    files = sorted(REPORT_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_equity(year: int) -> tuple[pd.DataFrame, Path | None]:
    path = latest_file(f"equity_{year}_*.csv") or latest_file("equity_*.csv")
    if not path:
        return pd.DataFrame(), None
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data.dropna(subset=["date"]), path


def load_signals(year: int) -> pd.DataFrame:
    path = latest_file(f"signals_{year}_*.csv") or latest_file("signals_*.csv")
    if not path:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"symbol": str}).fillna("")


def enrich_indicators(data: pd.DataFrame) -> pd.DataFrame:
    return calculate_signals(data)


def status_lights(is_buy: bool, is_holding: bool, is_sell: bool, drawdown: float) -> None:
    warning = drawdown >= 3.0 or is_sell
    safe = not warning and not is_buy and is_holding
    lights = [
        ("buy", "绿色", "买入机会", is_buy and not warning),
        ("safe", "蓝色", "安全持仓", safe),
        ("risk", "红色", "回撤预警 / 强平", warning),
    ]
    html = '<div class="signal-row">'
    for kind, color, label, active in lights:
        state = "active" if active else "idle"
        html += (
            f'<div class="signal-panel {kind} {state}">'
            f'<span class="signal-dot"></span><div><b>{color} · {label}</b>'
            f'<small>{"当前状态" if active else "待命"}</small></div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def drawdown_tank(drawdown: float, soft_limit: float, hard_limit: float) -> None:
    width = min(100.0, max(0.0, drawdown / hard_limit * 100.0)) if hard_limit else 0.0
    risk = drawdown >= 3.0
    state = "risk" if risk else "normal"
    if drawdown >= hard_limit:
        label = "硬熔断"
    elif drawdown >= soft_limit:
        label = "暂停开仓"
    elif risk:
        label = "提前预警"
    else:
        label = "安全区"
    html = f"""
    <div class="tank-wrap {state}">
      <div class="tank-head"><span>组合当前回撤</span><strong>{drawdown:.2f}% · {label}</strong></div>
      <div class="tank-track">
        <div class="tank-fill" style="width:{width:.2f}%"></div>
        <span class="marker m3">3%</span><span class="marker m5">5%</span><span class="marker m8">8%</span>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def make_chart(
    data: pd.DataFrame,
    title: str,
    period: str,
    turtle_dates: set[pd.Timestamp] | None = None,
    strategy: str = "double_high",
) -> go.Figure:
    if period.endswith("年"):
        year = int(period[:-1])
        display = data.loc[data["date"].dt.year == year].copy()
    else:
        display = data.tail(int(period.replace("日", ""))).copy()
    if display.empty:
        display = data.tail(120).copy()
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.76, 0.24])
    figure.add_trace(
        go.Candlestick(
            x=display["date"], open=display["open"], high=display["high"], low=display["low"], close=display["close"],
            name="K线", increasing_line_color="#d94841", decreasing_line_color="#198754",
        ), row=1, col=1,
    )
    figure.add_trace(go.Scatter(x=display["date"], y=display["ma5"], name="MA5", line=dict(color="#d4a017", width=1.8)), row=1, col=1)
    figure.add_trace(go.Scatter(x=display["date"], y=display["boll_mid"], name="BOLL中轨", line=dict(color="#1971c2", width=1.8)), row=1, col=1)
    volume_colors = ["#d94841" if close >= opened else "#198754" for opened, close in zip(display["open"], display["close"])]
    figure.add_trace(go.Bar(x=display["date"], y=display["volume"], name="成交量", marker_color=volume_colors, opacity=0.68), row=2, col=1)
    if strategy == "double_high":
        raw_hits = display.loc[display["raw_double_high"] & (display["trade_action"] != "BUY")]
        figure.add_trace(
            go.Scatter(
                x=raw_hits["date"], y=raw_hits["high"] * 1.015, mode="markers", name="普通双高",
                marker=dict(symbol="diamond", size=7, color="#d4a017"),
                hovertext="普通双高：尚未通过全部买入过滤", hoverinfo="text+x+y",
            ), row=1, col=1,
        )
        for row in display.loc[display["trade_action"] == "BUY"].itertuples():
            figure.add_annotation(
                x=row.date, y=row.high * 1.04, text="🚀", hovertext=f"买入候选（次日执行）：{row.trade_reason}",
                showarrow=True, arrowhead=2, arrowsize=1.2,
                arrowcolor="#b8860b", font=dict(size=24, color="#b8860b"), bgcolor="rgba(255,255,255,0.8)", row=1, col=1,
            )
        for row in display.loc[display["trade_action"] == "SELL"].itertuples():
            figure.add_annotation(
                x=row.date, y=row.high * 1.04, text="卖", hovertext=f"卖出信号（次日执行）：{row.trade_reason}",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowcolor="#c92a2a",
                font=dict(size=13, color="#ffffff"), bgcolor="#c92a2a", bordercolor="#a61e1e", borderpad=4, row=1, col=1,
            )
    elif strategy == "turtle":
        turtle_hits = display.loc[display["date"].isin(turtle_dates or set())]
        if not turtle_hits.empty:
            figure.add_trace(
                go.Scatter(
                    x=turtle_hits["date"], y=turtle_hits["low"] * 0.96, mode="markers+text", name="海龟",
                    marker=dict(symbol="circle", size=16, color="#0b7285", line=dict(color="#ffffff", width=1)),
                    text=["🐢"] * len(turtle_hits), textposition="middle center", textfont=dict(size=13),
                    hovertext="正式海龟：指数大跌、个股逆势涨幅达标、20日量比确认", hoverinfo="text+x+y",
                ), row=1, col=1,
            )
    elif strategy == PULLBACK_STRATEGY:
        pullback_hits = display.loc[display["pullback_ignition"]]
        if not pullback_hits.empty:
            figure.add_trace(
                go.Scatter(
                    x=pullback_hits["date"], y=pullback_hits["high"] * 1.025,
                    mode="markers+text", name="回调起爆",
                    marker=dict(symbol="triangle-up", size=17, color="#e67700", line=dict(color="#ffffff", width=1)),
                    text=["起"] * len(pullback_hits), textposition="middle center",
                    textfont=dict(size=11, color="#ffffff"),
                    hovertext="回调起爆：强趋势缩量回调结束，放量大阳突破平台",
                    hoverinfo="text+x+y",
                ), row=1, col=1,
            )
    elif strategy == LIFECYCLE_STRATEGY:
        lifecycle_hits = display.loc[display["lifecycle_buy"]]
        if not lifecycle_hits.empty:
            figure.add_trace(
                go.Scatter(
                    x=lifecycle_hits["date"], y=lifecycle_hits["high"] * 1.03,
                    mode="markers+text", name=LIFECYCLE_LABEL,
                    marker=dict(symbol="star", size=16, color="#d9480f", line=dict(color="#ffffff", width=1)),
                    text=["女"] * len(lifecycle_hits), textposition="middle center",
                    textfont=dict(size=10, color="#ffffff"),
                    hovertext="女人第2交易日双高：次日开盘模拟介入",
                    hoverinfo="text+x+y",
                ), row=1, col=1,
            )
        # Crossing the upper band is not an initial entry.  It is only a
        # pyramiding observation after the stock has reached goddess stage;
        # the paper engine still requires an existing lifecycle position.
        upper_hits = display.loc[display["upper_break"] & display["goddess"]]
        if not upper_hits.empty:
            figure.add_trace(
                go.Scatter(
                    x=upper_hits["date"], y=upper_hits["high"] * 1.01,
                    mode="markers", name="女神上轨观察（非首买）",
                    marker=dict(
                        symbol="triangle-up-open", size=10, color="#f08c00",
                        line=dict(color="#f08c00", width=2),
                    ),
                    hovertext="非首买信号：仅已有女鬼周期持仓且风控允许时，次日模拟加仓5%",
                    hoverinfo="text+x+y",
                ), row=1, col=1,
            )
    figure.update_layout(
        title=dict(text=title, font=dict(size=18)), height=620, margin=dict(l=18, r=18, t=52, b=18),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=True, gridcolor="#edf0f2", zeroline=False)
    return figure


st.markdown(
    """
    <style>
      .stApp { background:#f5f7f8; color:#1c252b; }
      [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu { display:none !important; }
      .block-container { max-width:1480px; padding-top:1.1rem; padding-bottom:2rem; }
      h1,h2,h3,p { letter-spacing:0 !important; }
      .app-head { display:flex; justify-content:space-between; align-items:end; border-bottom:1px solid #ccd3d8; padding-bottom:12px; margin-bottom:14px; }
      .app-head h1 { font-size:27px; margin:0; }
      .app-head span { color:#5d6971; font-size:13px; }
      .signal-row { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:8px 0 16px; }
      .signal-panel { height:82px; display:flex; align-items:center; gap:15px; border:1px solid #cfd6da; border-radius:6px; padding:0 20px; background:#fff; opacity:.48; }
      .signal-panel.active { opacity:1; border-width:2px; box-shadow:0 3px 12px rgba(20,30,36,.10); }
      .signal-dot { width:30px; height:30px; border-radius:50%; flex:0 0 30px; background:#adb5bd; box-shadow:inset 0 0 0 5px rgba(255,255,255,.65); }
      .signal-panel.buy.active { border-color:#198754; color:#12643f; } .signal-panel.buy.active .signal-dot { background:#20a464; box-shadow:0 0 18px #20a464; }
      .signal-panel.safe.active { border-color:#1971c2; color:#145b9d; } .signal-panel.safe.active .signal-dot { background:#228be6; box-shadow:0 0 18px #228be6; }
      .signal-panel.risk.active { border-color:#c92a2a; color:#a61e1e; } .signal-panel.risk.active .signal-dot { background:#e03131; box-shadow:0 0 18px #e03131; animation:pulse 1s infinite; }
      .signal-panel b { display:block; font-size:18px; } .signal-panel small { display:block; margin-top:4px; color:#68747c; }
      .tank-wrap { background:#fff; border:1px solid #cfd6da; border-radius:6px; padding:16px 18px 24px; margin-bottom:14px; }
      .tank-wrap.risk { border-color:#c92a2a; animation:risk-border 1s infinite; }
      .tank-head { display:flex; justify-content:space-between; margin-bottom:12px; font-size:16px; }
      .tank-track { height:30px; position:relative; background:#e9ecef; border:1px solid #adb5bd; overflow:visible; }
      .tank-fill { height:100%; background:#20a464; transition:width .6s ease; }
      .tank-wrap.risk .tank-fill { background:#e03131; animation:pulse-fill 1s infinite; }
      .marker { position:absolute; top:34px; font-size:11px; color:#59636a; transform:translateX(-50%); }
      .m3 { left:37.5%; } .m5 { left:62.5%; } .m8 { left:99%; }
      div[data-testid="stMetric"] { background:#fff; border:1px solid #d4dade; border-radius:6px; padding:10px 14px; }
      div[data-testid="stMetric"] label { color:#59636a; }
      @keyframes pulse { 50% { opacity:.35; transform:scale(.88); } }
      @keyframes pulse-fill { 50% { background:#a61e1e; } }
      @keyframes risk-border { 50% { box-shadow:0 0 0 3px rgba(224,49,49,.17); } }
      @media(max-width:800px){ .signal-row{grid-template-columns:1fr;} .app-head{align-items:start;gap:8px;flex-direction:column;} }
    </style>
    """,
    unsafe_allow_html=True,
)


if not PRICE_PATH.exists() or not WATCHLIST_PATH.exists():
    st.error("缺少真实行情或自选股数据，请先在桌面系统中更新行情。")
    st.stop()

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
market_info = json.loads(MARKET_REPORT.read_text(encoding="utf-8")) if MARKET_REPORT.exists() else {}
prices = load_prices(str(PRICE_PATH), PRICE_PATH.stat().st_mtime)
watchlist = load_watchlist(str(WATCHLIST_PATH), WATCHLIST_PATH.stat().st_mtime)
display_year = int(prices["date"].max().year)
equity, equity_path = load_equity(display_year)
signals = load_signals(display_year)
paper_state = load_paper_account(PAPER_PATHS)
if paper_state is None:
    paper_state = initialize_paper_account(
        PAPER_PATHS,
        config,
        prices["date"].max().strftime("%Y-%m-%d"),
    )
turtle_events = (
    load_analytics_csv(str(TURTLE_EVENTS_PATH), TURTLE_EVENTS_PATH.stat().st_mtime)
    if TURTLE_EVENTS_PATH.exists() else pd.DataFrame()
)
pullback_scan = (
    load_signal_scan(str(PULLBACK_SCAN_PATH), PULLBACK_SCAN_PATH.stat().st_mtime)
    if PULLBACK_SCAN_PATH.exists() else pd.DataFrame()
)
lifecycle_scan = (
    load_signal_scan(str(LIFECYCLE_SCAN_PATH), LIFECYCLE_SCAN_PATH.stat().st_mtime)
    if LIFECYCLE_SCAN_PATH.exists() else pd.DataFrame()
)

available = set(prices["symbol"].unique())
options = watchlist.loc[watchlist["code"].isin(available)].copy()
options["label"] = options.apply(lambda row: f"{row['code']}  {row['name'] if row['name'] and row['name'] != row['code'] else ''}".strip(), axis=1)
label_to_code = dict(zip(options["label"], options["code"]))
all_options = watchlist.copy()
all_options["label"] = all_options.apply(
    lambda row: f"{row['code']}  {row['name'] if row['name'] and row['name'] != row['code'] else ''}".strip(),
    axis=1,
)
all_label_to_code = dict(zip(all_options["label"], all_options["code"]))

st.markdown(
    f'<div class="app-head"><h1>量化回撤动态看板</h1><span>系统日期 {datetime.now():%Y-%m-%d} · 完整行情截至 {market_info.get("latest_date", "--")} · '
    f'{market_info.get("succeeded", 0)}/{market_info.get("requested", 0)}只 · 5%-8%风控</span></div>',
    unsafe_allow_html=True,
)

# This entry point is deliberately outside the tabs: every subsystem can use
# the same workflow when a code is missing from the system watchlist.
st.markdown("##### 新增股票代码 · 先加入系统自选，再更新行情")
global_code_col, global_name_col, global_add_col = st.columns([1.4, 2.2, 1])
with global_code_col:
    global_code = st.text_input(
        "6位A股代码",
        placeholder="例如 000598 或 300483",
        key="global_new_watch_code",
        label_visibility="collapsed",
    ).strip()
with global_name_col:
    global_name = st.text_input(
        "股票名称（可不填）",
        placeholder="名称可留空，联网更新时自动补齐",
        key="global_new_watch_name",
        label_visibility="collapsed",
    ).strip()
with global_add_col:
    global_add_clicked = st.button("加入系统自选", type="primary", width="stretch", key="global_add_watch")

watch_codes = set(watchlist["code"].astype(str).str.zfill(6))
if global_code:
    if not (len(global_code) == 6 and global_code.isdigit() and SUPPORTED_STOCK_CODE.fullmatch(global_code)):
        st.error("请输入受支持的6位沪深京A股代码，例如 000598 或 300483。")
    elif global_code in watch_codes:
        if global_code not in available:
            st.warning(f"{global_code} 已在系统自选，但本地还没有有效日线；点击“通达信联网更新”后再判断信号。")
        else:
            latest_code_date = prices.loc[prices["symbol"] == global_code, "date"].max()
            st.info(f"{global_code} 已在系统自选，行情截至 {latest_code_date:%Y-%m-%d}。")
    else:
        st.info(f"{global_code} 尚未加入系统自选。点击右侧按钮后，双高、海龟、回调起爆和女鬼周期都会纳入扫描。")
        if global_add_clicked:
            item, added = add_watch_item(WATCHLIST_PATH, global_code, global_name)
            st.session_state["direct_watch_flash"] = (
                f"已新增 {item.code} {item.name} 到系统自选。请点击“通达信联网更新”补齐日线后再看信号。"
                if added else f"{item.code} 已在系统自选中。"
            )
            st.cache_data.clear()
            st.rerun()

    if global_code in available:
        diagnostic_stock = enrich_indicators(prices.loc[prices["symbol"] == global_code])
        diagnostic_dashboard = calculate_signals(diagnostic_stock)
        diagnostic_lifecycle = calculate_lifecycle_signals(
            diagnostic_stock,
            lifecycle_rules_from_config(config),
        )
        diagnostic_row = diagnostic_dashboard.iloc[-1]
        lifecycle_row = diagnostic_lifecycle.iloc[-1]
        diag_cols = st.columns(6)
        diag_cols[0].metric("普通双高", "是" if bool(diagnostic_row["raw_double_high"]) else "否")
        diag_cols[1].metric("火箭买入", "是" if bool(diagnostic_row["buy_candidate"]) else "否")
        diag_cols[2].metric("女鬼变女人", "是" if bool(lifecycle_row["woman_day1"]) else "否")
        diag_cols[3].metric("女鬼周期介入", "是" if bool(lifecycle_row["eligible"]) else "否")
        diag_cols[4].metric("女神阶段", "是" if bool(lifecycle_row["goddess"]) else "否")
        diag_cols[5].metric("热点/妖股风控", "是" if bool(lifecycle_row["hot_stock"]) else "否")
        if bool(diagnostic_row["buy_candidate"]):
            diagnostic_action = "双高火箭介入候选：收盘确认，下一交易日仅按模拟账户规则执行。"
        elif bool(lifecycle_row["eligible"]):
            diagnostic_action = "女鬼周期介入候选：收盘确认，下一交易日仅按模拟账户规则执行。"
        elif bool(lifecycle_row["woman_day1"]):
            diagnostic_action = "女人第1日观察：今天不介入，等待下一交易日双高和量能确认。"
        elif bool(lifecycle_row["hot_stock"]) and float(lifecycle_row["close"]) < float(lifecycle_row["ma5"]):
            diagnostic_action = "热点/妖股跌破MA5：按生命周期持仓规则触发离场；空仓不接。"
        elif bool(lifecycle_row["hot_stock"]):
            diagnostic_action = "热点/妖股风控观察：空仓不追；已有持仓仅在MA5上方按规则跟踪。"
        elif bool(diagnostic_row["raw_double_high"]):
            diagnostic_action = "普通双高：只观察，尚未通过火箭介入过滤。"
        else:
            diagnostic_action = "当前无正式介入或离场信号，继续观察。"
        st.warning(f"系统动作：{diagnostic_action}")
        st.caption(
            f"{global_code} 最新完整日线 {pd.Timestamp(diagnostic_row['date']):%Y-%m-%d}："
            f"收盘 {float(diagnostic_row['close']):.2f}，MA5 {float(diagnostic_row['ma5']):.2f}，"
            f"MA20 {float(diagnostic_row['ma20']):.2f}，量比20 {float(diagnostic_row['volume_ratio']):.2f}。"
            f"这是收盘确认结果，不能把盘中价当成已完成信号。"
        )

if "market_sync_flash" in st.session_state:
    st.success(st.session_state.pop("market_sync_flash"))

control_a, control_b, control_c, control_d = st.columns([2.2, 1, 1, 1.2])
with control_a:
    selected_label = st.selectbox("股票", list(label_to_code), index=0)
with control_b:
    chart_period = st.segmented_control("K线区间", options=[f"{display_year}年", "60日", "120日", "250日"], default=f"{display_year}年")
with control_c:
    if st.button("重读本地数据", width="stretch", type="secondary"):
        st.cache_data.clear()
        st.rerun()
with control_d:
    if st.button(
        "通达信联网更新",
        width="stretch",
        type="primary",
        help=f"当前只纳入截至 {latest_completed_market_date()} 的完整日K",
    ):
        progress_bar = st.progress(0.0, text="正在连接通达信行情节点...")

        def update_progress(done: int, total: int, message: str) -> None:
            progress_bar.progress(min(1.0, done / max(1, total)), text=message)

        try:
            update_result = update_watchlist_prices(
                load_watch_items(WATCHLIST_PATH),
                PRICE_PATH,
                MARKET_REPORT,
                days=800,
                progress=update_progress,
            )
        except Exception as error:
            progress_bar.empty()
            st.error(f"联网更新失败，原行情文件保持不变：{error}")
        else:
            progress_bar.progress(0.97, text="行情已更新，正在补齐通达信证券名称...")
            name_result = refresh_watchlist_names(
                load_watch_items(WATCHLIST_PATH),
                WATCHLIST_PATH,
                progress=lambda done, total, message: progress_bar.progress(0.97, text=message),
            )
            progress_bar.progress(0.98, text="名称已同步，正在更新四个基准指数...")
            benchmark_result = update_benchmark_prices(
                BENCHMARK_PATH,
                BENCHMARK_REPORT,
                days=800,
            )
            progress_bar.progress(0.99, text="指数已更新，正在重扫双高、海龟、女鬼周期、回调起爆和涨停助手...")
            updated_prices = load_prices(str(PRICE_PATH), PRICE_PATH.stat().st_mtime)
            updated_watchlist = load_watchlist(str(WATCHLIST_PATH), WATCHLIST_PATH.stat().st_mtime)
            scan_watchlist_signals(updated_prices, updated_watchlist, SIGNAL_SCAN_PATH, 10)
            updated_benchmarks = load_benchmarks(str(BENCHMARK_PATH), BENCHMARK_PATH.stat().st_mtime)
            turtle_result = analyze_turtle_events(
                updated_prices,
                updated_benchmarks,
                config,
                names=dict(zip(updated_watchlist["code"], updated_watchlist["name"])),
            )
            save_turtle_bundle(turtle_result, TURTLE_ANALYTICS_DIR)
            turtle_reversal_result = analyze_turtle_reversal(updated_prices, turtle_result.events, config)
            save_turtle_reversal_bundle(turtle_reversal_result, TURTLE_REVERSAL_DIR)
            pullback_result = scan_pullback_signals(
                updated_prices,
                updated_watchlist,
                PULLBACK_SCAN_PATH,
                config,
                recent_days=800,
            )
            lifecycle_result = scan_lifecycle_signals(
                updated_prices,
                updated_watchlist,
                LIFECYCLE_SCAN_PATH,
                config,
                recent_days=800,
            )
            limit_assistant_result = scan_limit_assistant(
                updated_prices,
                updated_watchlist,
                recent_trading_days=60,
                output_path=LIMIT_ASSISTANT_SCAN_PATH,
            )
            progress_bar.progress(1.0, text="行情和信号更新完成")
            st.session_state["market_sync_flash"] = (
                f"通达信联网更新完成：{update_result.succeeded}/{update_result.requested}只，"
                f"最近完整交易日 {update_result.latest_date}；补齐{name_result.updated}个名称，"
                f"{benchmark_result.succeeded}个指数已更新，海龟事件{len(turtle_result.events)}条；"
                f"双高/火箭已重扫，女鬼周期{len(lifecycle_result)}条，回调起爆{len(pullback_result)}条，"
                f"涨停助手{len(limit_assistant_result)}条；失败股票保留原有数据。"
            )
            st.cache_data.clear()
            st.rerun()

code = label_to_code[selected_label]
stock_name = options.loc[options["code"] == code, "name"].iloc[0]
stock = enrich_indicators(prices.loc[prices["symbol"] == code])
latest = stock.iloc[-1]

current_drawdown = float(equity.iloc[-1]["drawdown_pct"]) if not equity.empty else 0.0
max_drawdown = float(equity["drawdown_pct"].max()) if not equity.empty else 0.0
latest_action = str(latest["trade_action"])
is_buy_candidate = latest_action == "BUY"
is_holding = bool(latest["holding_state"])
is_sell = latest_action == "SELL"
buy_authorized = is_buy_candidate and current_drawdown < float(config["soft_drawdown_pct"])
status_lights(buy_authorized, is_holding, is_sell, current_drawdown)
drawdown_tank(current_drawdown, float(config["soft_drawdown_pct"]), float(config["hard_drawdown_pct"]))

metric_cols = st.columns(6)
metric_cols[0].metric("组合权益", f"{float(equity.iloc[-1]['equity']):,.2f}" if not equity.empty else "--")
metric_cols[1].metric("当前回撤", f"{current_drawdown:.2f}%")
metric_cols[2].metric("最大回撤", f"{max_drawdown:.2f}%")
metric_cols[3].metric("最新收盘", f"{latest['close']:.2f}", f"{(latest['close']/stock.iloc[-2]['close']-1)*100:.2f}%")
if is_sell:
    stock_signal = "卖出"
elif is_buy_candidate:
    stock_signal = "买入候选"
elif is_holding:
    stock_signal = "持有"
else:
    stock_signal = "观察"
if current_drawdown >= float(config["hard_drawdown_pct"]):
    portfolio_permission = "熔断禁买"
elif current_drawdown >= float(config["soft_drawdown_pct"]):
    portfolio_permission = "暂停开仓"
else:
    portfolio_permission = "允许执行"
metric_cols[4].metric("个股信号", stock_signal)
metric_cols[5].metric("组合许可", portfolio_permission)

chart_tab, search_tab, client_tab, scan_tab, analytics_tab, turtle_tab, taozi_tab, pullback_tab, lifecycle_tab, limit_tab, candidate03_tab, risk_tab, paper_tab = st.tabs(
    ["K线与双高", "搜股与模拟买入", "客户端同步", "自选股信号扫描", "信号统计", "海龟研究", "质量趋势精选", "回调起爆", "女鬼周期", "涨停助手研究", "候选03观察", "风控事件", "模拟账户"]
)
with chart_tab:
    year_rows = stock.loc[stock["date"].dt.year == display_year]
    date_span = f"{year_rows['date'].min():%Y-%m-%d} 至 {year_rows['date'].max():%Y-%m-%d}" if not year_rows.empty else ""
    title = f"{code} {stock_name if stock_name != code else ''} · MA5 / BOLL20中轨 · {date_span}"
    st.markdown("**金色菱形**＝普通双高；**🚀**＝通过双高买入过滤；**卖**＝双高策略的止损、移动保护或趋势破坏。本图不混入海龟策略。信号在收盘确认，按下一交易日执行。")
    st.plotly_chart(make_chart(stock, title, chart_period or f"{display_year}年"), width="stretch", config={"displaylogo": False, "scrollZoom": True})
    hits = stock.loc[stock["trade_action"] != "", ["date", "trade_action", "signal_score", "close", "volume_ratio", "trade_reason"]].tail(20).copy()
    if not hits.empty:
        hits["date"] = hits["date"].dt.strftime("%Y-%m-%d")
        hits["volume_ratio"] = hits["volume_ratio"].round(2)
        st.dataframe(
            hits.rename(columns={"date": "日期", "trade_action": "信号", "signal_score": "评分", "close": "收盘", "volume_ratio": "量比20", "trade_reason": "原因"}),
            hide_index=True, width="stretch",
        )
with limit_tab:
    st.markdown("#### 涨停助手研究 · 首板质量与次日候选")
    st.caption(
        "把同花顺涨停助手中可验证的日线信息落地为独立策略：首板、连板、冲板未封、"
        "20日突破、趋势、量比和收盘强度。盘口逐笔、封单金额、主力净量属于Level-2，"
        "未导入同花顺快照时不会虚构。信号收盘确认，按A股T+1下一交易日观察。"
    )
    if "limit_assistant_flash" in st.session_state:
        st.success(st.session_state.pop("limit_assistant_flash"))
    la_a, la_b, la_c = st.columns([1.4, 1.4, 1])
    with la_a:
        la_days = st.number_input("显示最近交易日", min_value=1, max_value=60, value=10, step=1, key="la_days")
    with la_b:
        la_filter = st.multiselect(
            "显示事件",
            ["首板", "连板", "涨停", "冲板未封"],
            default=["首板", "连板", "冲板未封"],
            key="la_filter",
        )
    with la_c:
        la_run = st.button("重算涨停助手", type="primary", width="stretch")
    if la_run:
        with st.spinner("正在用本地日线计算涨停异动和首板质量..."):
            la_scan = scan_limit_assistant(
                prices,
                watchlist,
                recent_trading_days=int(la_days),
                output_path=LIMIT_ASSISTANT_SCAN_PATH,
            )
        st.session_state["limit_assistant_scan"] = la_scan
        st.session_state["limit_assistant_flash"] = f"涨停助手研究完成：识别 {len(la_scan)} 条日线异动。"
        st.rerun()
    la_scan = st.session_state.get("limit_assistant_scan")
    if la_scan is None and LIMIT_ASSISTANT_SCAN_PATH.exists():
        try:
            la_scan = pd.read_csv(LIMIT_ASSISTANT_SCAN_PATH, encoding="utf-8-sig")
            if "date" in la_scan:
                la_scan["date"] = pd.to_datetime(la_scan["date"], errors="coerce")
            st.session_state["limit_assistant_scan"] = la_scan
        except (OSError, pd.errors.ParserError):
            la_scan = None
    if la_scan is None or la_scan.empty:
        st.info("点击“重算涨停助手”生成最近异动；该页不会自动提交实盘订单。")
    else:
        shown = la_scan.copy()
        if la_filter:
            shown = shown.loc[shown["event_type"].isin(la_filter)]
        latest_la_date = pd.to_datetime(la_scan["date"], errors="coerce").max()
        la_metrics = st.columns(5)
        la_metrics[0].metric("异动事件", len(la_scan))
        la_metrics[1].metric("首板", int((la_scan["event_type"] == "首板").sum()))
        la_metrics[2].metric("可观察候选", int(la_scan["buy_candidate"].astype(bool).sum()))
        la_metrics[3].metric("最新日期", latest_la_date.strftime("%Y-%m-%d") if pd.notna(latest_la_date) else "--")
        la_metrics[4].metric("策略定位", "研究/观察")
        display_cols = {
            "date": "日期", "symbol": "代码", "name": "名称", "event_type": "异动",
            "change_pct": "涨幅%", "volume_ratio": "量比20", "signal_score": "评分",
            "buy_candidate": "首板候选", "reason": "判断",
        }
        if shown.empty:
            st.info("当前筛选条件下没有事件。")
        else:
            shown = shown[list(display_cols)].rename(columns=display_cols)
            shown["日期"] = pd.to_datetime(shown["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
            st.dataframe(shown, hide_index=True, width="stretch")
        st.markdown("##### 研究口径")
        st.write(
            "首板候选必须同时满足：非ST、趋势多头（收盘价>MA20>MA60）、"
            "收盘突破20日高点、量比20为1.10至4.00、收盘位于当日振幅上部且上影线短、"
            "距MA20不超过18%。它是候选，不是保证涨停或买入指令。"
        )
        if st.button("计算首板候选前向收益", width="stretch"):
            with st.spinner("正在计算1/3/5/10个交易日的前向收益..."):
                la_returns = event_forward_returns(prices)
            st.session_state["limit_assistant_returns"] = la_returns
        la_returns = st.session_state.get("limit_assistant_returns")
        if la_returns is not None:
            if la_returns.empty:
                st.warning("当前历史行情中没有满足全部条件且可计算后续收益的样本。")
            else:
                ret_metrics = st.columns(5)
                ret_metrics[0].metric("样本数", len(la_returns))
                for metric_col, h in zip(ret_metrics[1:], (1, 3, 5, 10)):
                    values = pd.to_numeric(la_returns[f"ret_{h}d_pct"], errors="coerce").dropna()
                    metric_col.metric(f"{h}日均收益", f"{values.mean():.2f}%" if not values.empty else "--")
                st.dataframe(la_returns.tail(100), hide_index=True, width="stretch")

with search_tab:
    st.markdown("#### 搜股票 → 加模拟自选 → 看信号 → 模拟买入")
    st.caption("可直接录入新的6位A股代码加入系统总自选；有真实行情后才能加入模拟自选并判断信号。")
    if "direct_watch_flash" in st.session_state:
        st.success(st.session_state.pop("direct_watch_flash"))
    search_a, search_b = st.columns([3, 1])
    with search_a:
        stock_keyword = st.text_input("股票代码或名称", placeholder="例如：300608 或 思特奇", key="paper_stock_keyword")
    search_matches = all_options.copy()
    if stock_keyword.strip():
        keyword = stock_keyword.strip().lower()
        searchable = search_matches[["code", "name"]].astype(str).agg(" ".join, axis=1).str.lower()
        search_matches = search_matches.loc[searchable.str.contains(keyword, regex=False)]
    search_matches = search_matches.head(50)
    match_labels = search_matches["label"].tolist()
    with search_b:
        match_label = st.selectbox(
            "搜索结果",
            match_labels,
            index=0 if match_labels else None,
            placeholder="没有匹配股票",
            key="paper_stock_match",
        )
    direct_code = stock_keyword.strip()
    code_already_exists = direct_code in set(all_options["code"])
    if direct_code and len(direct_code) == 6 and direct_code.isdigit() and not code_already_exists:
        if SUPPORTED_STOCK_CODE.fullmatch(direct_code):
            direct_a, direct_b = st.columns([3, 1])
            with direct_a:
                direct_name = st.text_input(
                    "股票名称（可不填）",
                    placeholder="不填时先用股票代码，更新行情后仍可补充名称",
                    key="direct_watch_name",
                )
                st.info(f"代码 {direct_code} 尚未收录，可以直接加入系统总自选。")
            with direct_b:
                st.write("")
                if st.button("直接加入总自选", type="primary", width="stretch"):
                    item, added = add_watch_item(WATCHLIST_PATH, direct_code, direct_name)
                    st.session_state["direct_watch_flash"] = (
                        f"已新增 {item.code} {item.name}。请在收盘后更新真实行情，再进行双高判断和模拟买入。"
                        if added else f"{item.code} 已在系统总自选中。"
                    )
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.error("该6位代码不属于当前支持的沪深京A股代码范围。")
    if match_label:
        match_code = all_label_to_code[match_label]
        match_name = all_options.loc[all_options["code"] == match_code, "name"].iloc[0]
        has_market_data = match_code in available
        already_watched = any(item["symbol"] == match_code for item in paper_state.get("paper_watchlist", []))
        add_a, add_b = st.columns([3, 1])
        with add_a:
            st.write(f"已选：**{match_code} {match_name if match_name != match_code else ''}**")
            if not has_market_data:
                st.warning("已加入系统总自选，但还没有有效日线。请收盘后更新真实行情。")
        with add_b:
            add_label = "待更新行情" if not has_market_data else ("已在模拟自选" if already_watched else "加入模拟自选")
            if st.button(
                add_label,
                width="stretch",
                disabled=already_watched or not has_market_data,
            ):
                add_paper_watch_symbol(PAPER_PATHS, match_code, match_name)
                st.rerun()

    paper_state = load_paper_account(PAPER_PATHS) or paper_state
    paper_watchlist = paper_state.get("paper_watchlist", [])
    if not paper_watchlist:
        st.info("模拟自选尚为空。先搜索股票并点击“加入模拟自选”。")
    else:
        watched_labels = [f"{item['symbol']}  {item['name']}" for item in paper_watchlist]
        selected_watch_label = st.selectbox("模拟自选", watched_labels, key="paper_watched_stock")
        watched_index = watched_labels.index(selected_watch_label)
        watched_item = paper_watchlist[watched_index]
        watched_code = watched_item["symbol"]
        watched_name = watched_item["name"]
        watched_stock = enrich_indicators(prices.loc[prices["symbol"] == watched_code])
        if watched_stock.empty:
            st.error("该股票在当前行情文件中没有有效日线，请先更新真实行情。")
        else:
            watched_latest = watched_stock.iloc[-1]
            watched_date = watched_latest["date"].strftime("%Y-%m-%d")
            raw_double_high = bool(watched_latest["raw_double_high"])
            high_quality_buy = bool(watched_latest["buy_candidate"])
            if high_quality_buy:
                st.success(f"{watched_date}：🚀 高质量买入信号。趋势、首次突破、量价、K线质量和乖离均通过。")
            elif raw_double_high:
                st.warning(f"{watched_date}：◆ 普通双高，仅观察，尚未通过全部买入过滤。")
            else:
                st.info(f"{watched_date}：当前没有双高或买入信号。")

            signal_metrics = st.columns(5)
            signal_metrics[0].metric("最新收盘", f"{float(watched_latest['close']):.2f}")
            signal_metrics[1].metric("信号评分", f"{int(watched_latest['signal_score'])}")
            signal_metrics[2].metric("20日量比", f"{float(watched_latest['volume_ratio']):.2f}" if pd.notna(watched_latest["volume_ratio"]) else "--")
            signal_metrics[3].metric("距MA20", f"{float(watched_latest['extension_pct']):.2f}%" if pd.notna(watched_latest["extension_pct"]) else "--")
            signal_metrics[4].metric(
                "账户风险",
                PAPER_RISK_LABELS.get(paper_state["risk_lock"]["status"], paper_state["risk_lock"]["status"]).split(" · ")[0],
            )

            account_equity = float(paper_state["equity_history"][-1]["equity"])
            maximum_value = min(
                float(paper_state["cash"]),
                account_equity * float(config["max_position_pct"]) / 100.0,
            )
            default_value = min(10000.0, maximum_value)
            buy_a, buy_b, buy_c = st.columns([1.4, 1.8, 1])
            with buy_a:
                target_value = st.number_input(
                    "计划买入市值（元）",
                    min_value=100.0,
                    max_value=max(100.0, float(maximum_value)),
                    value=max(100.0, float(default_value)),
                    step=1000.0,
                    key=f"paper_target_{watched_code}",
                )
                st.caption(f"当前单股风控上限：{maximum_value:,.2f}元")
            with buy_b:
                simulation_confirmed = st.checkbox(
                    "确认仅为模拟委托，不连接券商",
                    key=f"paper_buy_confirm_{watched_code}",
                )
                date_ready = paper_state["last_processed_date"] == watched_date
                if not date_ready:
                    st.caption("账户日期与行情不一致，请先在“模拟账户”页推进至最新行情。")
                elif watched_code in paper_state["pending_entries"]:
                    pending = paper_state["pending_entries"][watched_code]
                    st.caption(f"已有待成交单：计划市值 {float(pending.get('target_value', maximum_value)):,.2f}元")
            with buy_c:
                st.write("")
                can_queue = (
                    high_quality_buy
                    and paper_state["risk_lock"]["status"] == RISK_NORMAL
                    and watched_code not in paper_state["positions"]
                    and watched_code not in paper_state["pending_entries"]
                    and paper_state["last_processed_date"] == watched_date
                )
                if st.button("提交模拟买入", type="primary", width="stretch", disabled=not can_queue):
                    rows = [
                        Candle(
                            row.date.strftime("%Y-%m-%d"), watched_code, float(row.open), float(row.high),
                            float(row.low), float(row.close), float(row.volume),
                        )
                        for row in watched_stock.itertuples()
                    ]
                    try:
                        queue_paper_buy_signal(
                            PAPER_PATHS,
                            {watched_code: rows},
                            config,
                            watched_code,
                            float(target_value),
                            simulation_confirmed,
                        )
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        st.rerun()
            if not high_quality_buy:
                st.caption("“提交模拟买入”保持禁用：普通双高不等于买点，必须等当前交易日出现高质量火箭信号。")
            if st.button("移出模拟自选", key=f"remove_paper_watch_{watched_code}"):
                remove_paper_watch_symbol(PAPER_PATHS, watched_code)
                st.rerun()
with client_tab:
    st.markdown("#### 通达信 / 同花顺客户端同步")
    st.caption("只同步自选股文件和只读行情，不读取账号密码，不控制交易进程，不发送真实订单。")
    sources = discover_client_sources()
    tdx_codes = parse_tdx_watchlist(sources.tdx_watchlist) if sources.tdx_watchlist else []
    ths_codes = parse_ths_watchlist(sources.ths_watchlist) if sources.ths_watchlist else []
    tdx_running = client_process_running("TdxW.exe")
    ths_running = client_process_running("hexin.exe")
    client_metrics = st.columns(4)
    client_metrics[0].metric("通达信客户端", "已检测" if sources.tdx_root else "未找到")
    client_metrics[1].metric("通达信有效自选", len(tdx_codes))
    client_metrics[2].metric("同花顺客户端", "运行中" if ths_running else ("已检测" if sources.ths_root else "未找到"))
    client_metrics[3].metric("同花顺有效自选", len(ths_codes))

    if "client_sync_flash" in st.session_state:
        st.success(st.session_state.pop("client_sync_flash"))

    import_tdx, import_ths = st.columns(2)
    with import_tdx:
        st.markdown("**从通达信导入系统**")
        st.code(str(sources.tdx_watchlist or "未找到自选文件"), language=None)
        if sources.tdx_root and not tdx_codes:
            st.info("当前通达信 `zxg.blk` 只有占位码，还没有有效A股自选。可先把系统自选导出到通达信。")
        if st.button("导入通达信自选", width="stretch", disabled=not bool(tdx_codes)):
            result = import_client_watchlist("通达信客户端", sources.tdx_watchlist, WATCHLIST_PATH)
            st.session_state["client_sync_flash"] = (
                f"通达信读取{result.source_count}只，新增{result.added_count}只，系统现有{result.after_count}只。"
            )
            st.cache_data.clear()
            st.rerun()
    with import_ths:
        st.markdown("**从同花顺只读导入**")
        st.code(str(sources.ths_watchlist or "未找到自选文件"), language=None)
        st.caption("即使同花顺正在运行也只读取自选JSON；不会写入同花顺，更不会接触下单模块。")
        if st.button("导入同花顺自选", width="stretch", disabled=not bool(ths_codes)):
            result = import_client_watchlist("同花顺客户端", sources.ths_watchlist, WATCHLIST_PATH)
            st.session_state["client_sync_flash"] = (
                f"同花顺读取{result.source_count}只，新增{result.added_count}只，系统现有{result.after_count}只。"
            )
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.markdown("**导出到通达信客户端**")
    export_a, export_b, export_c = st.columns([1.3, 2, 1])
    with export_a:
        export_scope = st.radio(
            "导出范围",
            ["模拟自选", "系统总自选"],
            horizontal=True,
            key="tdx_export_scope",
        )
    paper_codes = [item["symbol"] for item in paper_state.get("paper_watchlist", [])]
    export_codes = paper_codes if export_scope == "模拟自选" else watchlist["code"].tolist()
    with export_b:
        tdx_closed_confirmed = st.checkbox("确认通达信客户端已关闭，并允许备份后写入自选文件", key="tdx_closed_confirmed")
        if tdx_running:
            st.error("检测到通达信仍在运行，请退出客户端后再导出。")
        else:
            st.caption(f"准备导出{len(export_codes)}只；原文件会备份到 `{CLIENT_BACKUP_DIR}`。")
    with export_c:
        st.write("")
        can_export_tdx = bool(sources.tdx_root and sources.tdx_watchlist and export_codes and not tdx_running)
        if st.button("写入通达信自选", type="primary", width="stretch", disabled=not can_export_tdx):
            try:
                export_result = export_tdx_watchlist(
                    export_codes,
                    sources.tdx_watchlist,
                    CLIENT_BACKUP_DIR,
                    tdx_closed_confirmed,
                )
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                st.success(
                    f"已写入{export_result['exported']}只；通达信自选合并后共{export_result['after']}只。重新打开通达信即可查看。"
                )

    st.info("通达信本地 `vipdoc` 日线目录当前为空，因此行情仍使用已接入的通达信只读行情协议；以后客户端下载本地日线后可再增加本地文件优先读取。")
with scan_tab:
    scan_head_a, scan_head_b = st.columns([3, 1])
    with scan_head_a:
        st.markdown("#### 当前自选股菱形 / 火箭")
    with scan_head_b:
        if st.button(f"重新扫描{len(watchlist)}只", width="stretch"):
            with st.spinner("正在计算最近10个交易日信号..."):
                scan_watchlist_signals(prices, watchlist, SIGNAL_SCAN_PATH, 10)
            st.cache_data.clear()
            st.rerun()
    if not SIGNAL_SCAN_PATH.exists():
        st.info("尚无扫描结果，请点击“重新扫描1003只”。")
    else:
        scan_data = load_signal_scan(str(SIGNAL_SCAN_PATH), SIGNAL_SCAN_PATH.stat().st_mtime)
        filter_a, filter_b, filter_c = st.columns([1.5, 1.1, 1.6])
        with filter_a:
            selected_types = st.multiselect(
                "信号类型", ["🚀 高质量买入候选", "◆ 普通双高"],
                default=["🚀 高质量买入候选", "◆ 普通双高"],
            )
        with filter_b:
            recent_days = st.segmented_control("时间范围", [1, 3, 5, 10], default=1, format_func=lambda value: "最新交易日" if value == 1 else f"最近{value}日")
        with filter_c:
            scan_keyword = st.text_input("搜索代码 / 名称 / 标签", placeholder="输入股票代码或名称")
        filtered = scan_data.loc[
            (scan_data["age_trade_days"] < int(recent_days or 1))
            & scan_data["signal"].isin(selected_types)
        ].copy()
        if scan_keyword.strip():
            keyword = scan_keyword.strip().lower()
            combined = filtered[["symbol", "name", "tags"]].astype(str).agg(" ".join, axis=1).str.lower()
            filtered = filtered.loc[combined.str.contains(keyword, regex=False)]
        current_date = prices["date"].max().strftime("%Y-%m-%d")
        rocket_count = int((filtered["signal"] == "🚀 高质量买入候选").sum())
        diamond_count = int((filtered["signal"] == "◆ 普通双高").sum())
        count_a, count_b, count_c, count_d = st.columns(4)
        count_a.metric("行情日期", current_date)
        count_b.metric("筛选结果", len(filtered))
        count_c.metric("🚀 火箭", rocket_count)
        count_d.metric("◆ 普通双高", diamond_count)
        if filtered.empty:
            st.info("当前筛选范围没有符合条件的信号。")
        else:
            permission = "允许候选" if current_drawdown < float(config["soft_drawdown_pct"]) else "组合暂停开仓"
            filtered["组合许可"] = filtered["signal"].map(lambda value: permission if value.startswith("🚀") else "仅观察")
            show = filtered[["date", "symbol", "name", "signal", "score", "close", "volume_ratio", "extension_pct", "组合许可", "reason", "tags"]].copy()
            show["date"] = show["date"].dt.strftime("%Y-%m-%d")
            show = show.rename(columns={
                "date": "日期", "symbol": "代码", "name": "名称", "signal": "信号", "score": "评分",
                "close": "收盘", "volume_ratio": "量比20", "extension_pct": "距MA20%", "reason": "判断原因", "tags": "自选标签",
            })
            st.dataframe(show, hide_index=True, width="stretch", height=min(620, 40 + len(show) * 35))
with analytics_tab:
    analytics_head_a, analytics_head_b = st.columns([3, 1])
    with analytics_head_a:
        st.markdown("#### 普通双高 / 火箭统计实验室")
        st.caption("信号收盘确认，次日开盘模拟买入；按100股整数手，计入佣金、印花税和双边滑点。")
    with analytics_head_b:
        if st.button("生成 / 刷新统计", type="primary", width="stretch"):
            analytics_progress = st.progress(0.0, text="正在构建市场状态...")

            def analytics_progress_callback(done: int, total: int, message: str) -> None:
                analytics_progress.progress(min(1.0, done / max(1, total)), text=message)

            try:
                analytics_result = analyze_signals(
                    prices,
                    config,
                    names=dict(zip(watchlist["code"], watchlist["name"])),
                    progress=analytics_progress_callback,
                )
                save_analytics_bundle(analytics_result, SIGNAL_ANALYTICS_DIR)
            except Exception as error:
                analytics_progress.empty()
                st.error(f"信号统计生成失败：{error}")
            else:
                analytics_progress.progress(1.0, text="统计报告生成完成")
                st.cache_data.clear()
                st.rerun()

    analytics_files = {
        "events": SIGNAL_ANALYTICS_DIR / "events_latest.csv",
        "summary": SIGNAL_ANALYTICS_DIR / "summary_latest.csv",
        "regime": SIGNAL_ANALYTICS_DIR / "regime_summary_latest.csv",
        "validation": SIGNAL_ANALYTICS_DIR / "validation_latest.csv",
        "walkforward": SIGNAL_ANALYTICS_DIR / "walkforward_latest.csv",
        "stability": SIGNAL_ANALYTICS_DIR / "stability_latest.csv",
        "metadata": SIGNAL_ANALYTICS_DIR / "metadata_latest.json",
        "report": SIGNAL_ANALYTICS_DIR / "report_latest.md",
    }
    analytics_ready = all(path.exists() for path in analytics_files.values())
    if not analytics_ready:
        st.info("尚未生成统计报告。点击“生成 / 刷新统计”，首次分析约需几十秒。")
    else:
        analytics_metadata = json.loads(analytics_files["metadata"].read_text(encoding="utf-8"))
        analytics_summary = load_analytics_csv(
            str(analytics_files["summary"]), analytics_files["summary"].stat().st_mtime
        )
        analytics_regime = load_analytics_csv(
            str(analytics_files["regime"]), analytics_files["regime"].stat().st_mtime
        )
        analytics_validation = load_analytics_csv(
            str(analytics_files["validation"]), analytics_files["validation"].stat().st_mtime
        )
        analytics_walkforward = load_analytics_csv(
            str(analytics_files["walkforward"]), analytics_files["walkforward"].stat().st_mtime
        )
        analytics_stability = load_analytics_csv(
            str(analytics_files["stability"]), analytics_files["stability"].stat().st_mtime
        )
        analytics_events = load_analytics_csv(
            str(analytics_files["events"]), analytics_files["events"].stat().st_mtime
        )
        current_price_end = prices["date"].max().strftime("%Y-%m-%d")
        if analytics_metadata.get("price_end") != current_price_end:
            st.warning(
                f"统计报告使用行情至 {analytics_metadata.get('price_end', '--')}，当前行情已到 {current_price_end}，请刷新统计。"
            )

        ten_day = analytics_summary.loc[analytics_summary["horizon_days"] == 10]
        ordinary_10 = ten_day.loc[ten_day["signal_type"] == "普通双高"]
        rocket_10 = ten_day.loc[ten_day["signal_type"] == "火箭"]

        def metric_value(frame: pd.DataFrame, column: str, suffix: str = "") -> str:
            return f"{float(frame.iloc[0][column]):.2f}{suffix}" if not frame.empty else "--"

        analytics_metrics = st.columns(6)
        analytics_metrics[0].metric("普通双高事件", int(analytics_metadata.get("ordinary_events", 0)))
        analytics_metrics[1].metric("火箭事件", int(analytics_metadata.get("rocket_events", 0)))
        analytics_metrics[2].metric("普通双高10日胜率", metric_value(ordinary_10, "win_rate_pct", "%"))
        analytics_metrics[3].metric("普通双高10日均值", metric_value(ordinary_10, "avg_return_pct", "%"))
        analytics_metrics[4].metric("火箭10日胜率", metric_value(rocket_10, "win_rate_pct", "%"))
        analytics_metrics[5].metric("火箭10日均值", metric_value(rocket_10, "avg_return_pct", "%"))
        st.caption(
            f"报告行情 {analytics_metadata.get('price_start')} 至 {analytics_metadata.get('price_end')} · "
            f"{analytics_metadata.get('symbols')}只 · 统一分析市值 {float(analytics_metadata.get('analysis_trade_value', 0)):,.2f}元 · "
            f"无法成交事件 {analytics_metadata.get('unfillable_events', 0)}条"
        )

        overview_tab, excursion_tab, regime_tab, validation_tab, stability_tab, detail_tab = st.tabs(
            ["收益与胜率", "有利/不利波动", "牛熊震荡", "样本外验证", "稳定性", "事件明细"]
        )
        with overview_tab:
            return_figure = go.Figure()
            for signal_type, color in (("普通双高", "#6c757d"), ("火箭", "#d4a017")):
                group = analytics_summary.loc[analytics_summary["signal_type"] == signal_type]
                return_figure.add_bar(
                    name=signal_type,
                    x=[f"{int(value)}日" for value in group["horizon_days"]],
                    y=group["avg_return_pct"],
                    marker_color=color,
                    text=[f"{value:.2f}%" for value in group["avg_return_pct"]],
                    textposition="outside",
                )
            return_figure.update_layout(
                title="计入交易成本后的平均收益",
                barmode="group",
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis_title="平均收益率 %",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(return_figure, width="stretch", config={"displaylogo": False})
            summary_show = analytics_summary.rename(
                columns={
                    "signal_type": "信号", "horizon_days": "持有日", "count": "样本数",
                    "win_rate_pct": "胜率%", "avg_return_pct": "平均收益%", "median_return_pct": "中位收益%",
                    "avg_gain_pct": "平均盈利%", "avg_loss_pct": "平均亏损%", "payoff_ratio": "盈亏比",
                    "best_return_pct": "最好%", "worst_return_pct": "最差%",
                }
            )
            st.dataframe(
                summary_show[["信号", "持有日", "样本数", "胜率%", "平均收益%", "中位收益%", "平均盈利%", "平均亏损%", "盈亏比", "最好%", "最差%"]],
                hide_index=True,
                width="stretch",
            )
        with excursion_tab:
            excursion = analytics_summary.loc[analytics_summary["horizon_days"] == 10].rename(
                columns={
                    "signal_type": "信号", "count": "10日样本数", "avg_mfe_pct": "平均最大有利波动%",
                    "avg_mae_pct": "平均最大不利波动%", "best_return_pct": "最佳10日收益%",
                    "worst_return_pct": "最差10日收益%",
                }
            )
            st.dataframe(
                excursion[["信号", "10日样本数", "平均最大有利波动%", "平均最大不利波动%", "最佳10日收益%", "最差10日收益%"]],
                hide_index=True,
                width="stretch",
            )
            st.caption("MFE表示持有期内最高价相对模拟买入价的最大顺向幅度；MAE表示最低价的最大逆向幅度。")
        with regime_tab:
            if analytics_regime.empty:
                st.info("暂无足够的市场状态分组样本。")
            else:
                regime_show = analytics_regime.rename(
                    columns={
                        "signal_type": "信号", "regime": "市场状态", "count": "10日样本数",
                        "win_rate_pct": "胜率%", "avg_return_pct": "平均收益%", "median_return_pct": "中位收益%",
                        "avg_gain_pct": "平均盈利%", "avg_loss_pct": "平均亏损%", "payoff_ratio": "盈亏比",
                        "avg_mfe_pct": "平均MFE%", "avg_mae_pct": "平均MAE%",
                    }
                )
                st.dataframe(
                    regime_show[["信号", "市场状态", "10日样本数", "胜率%", "平均收益%", "中位收益%", "平均盈利%", "平均亏损%", "盈亏比", "平均MFE%", "平均MAE%"]],
                    hide_index=True,
                    width="stretch",
                )
                st.caption("市场状态使用当前股票池60日收益中位数和站上MA20比例划分，不冒充官方大盘指数。")
        with validation_tab:
            if analytics_validation.empty:
                st.info("有效日期不足，暂时不能进行样本内外切分。")
            else:
                validation_show = analytics_validation.rename(
                    columns={
                        "signal_type": "信号", "sample": "样本", "split_date": "切分日", "count": "样本数",
                        "win_rate_pct": "胜率%", "avg_return_pct": "平均收益%", "median_return_pct": "中位收益%",
                        "avg_gain_pct": "平均盈利%", "avg_loss_pct": "平均亏损%", "payoff_ratio": "盈亏比",
                        "best_return_pct": "最好%", "worst_return_pct": "最差%",
                    }
                )
                st.dataframe(validation_show, hide_index=True, width="stretch")
                st.caption("按时间顺序以前60%作为样本内、后40%作为样本外；没有使用未来数据重新挑选参数。")
            st.markdown("**扩展窗口滚动验证**")
            if analytics_walkforward.empty:
                st.info("日期窗口不足，暂时不能进行滚动验证。")
            else:
                walkforward_show = analytics_walkforward.rename(
                    columns={
                        "signal_type": "信号", "fold": "窗口", "train_end": "训练截止",
                        "test_start": "测试开始", "test_end": "测试结束", "train_count": "训练样本",
                        "test_count": "测试样本", "train_avg_return_pct": "训练均值%",
                        "test_win_rate_pct": "测试胜率%", "test_avg_return_pct": "测试均值%",
                        "test_median_return_pct": "测试中位数%", "test_payoff_ratio": "测试盈亏比",
                        "direction_consistent": "训练测试同为正",
                    }
                )
                st.dataframe(walkforward_show, hide_index=True, width="stretch")
                st.caption("五段时间序列形成四个扩展窗口：每次只用当前窗口之前的历史，测试紧随其后的时间段。")
        with stability_tab:
            stability_show = analytics_stability.rename(
                columns={
                    "signal_type": "信号", "usable_10d_samples": "10日有效样本",
                    "train_avg_return_pct": "样本内均值%", "test_avg_return_pct": "样本外均值%",
                    "return_std_pct": "收益波动%", "positive_month_ratio_pct": "正收益月份占比%",
                    "months": "月份数", "positive_walkforward_ratio_pct": "正收益滚动窗口占比%",
                    "walkforward_folds": "滚动窗口数", "max_consecutive_losses": "最大连续亏损次数",
                    "overlap_ratio_pct": "10日重叠事件占比%", "assessment": "稳定性判断",
                }
            )
            st.dataframe(stability_show, hide_index=True, width="stretch")
            st.warning("“样本内外方向一致”只代表历史统计较一致，不等于未来保证盈利。重叠事件比例高时，有效独立样本会少于表面数量。")
        with detail_tab:
            detail_a, detail_b = st.columns(2)
            with detail_a:
                detail_types = st.multiselect("信号", ["普通双高", "火箭"], default=["普通双高", "火箭"])
            with detail_b:
                detail_regimes = st.multiselect("市场状态", ["牛市", "震荡市", "下跌市"], default=["牛市", "震荡市", "下跌市"])
            filtered_events = analytics_events.loc[
                analytics_events["signal_type"].isin(detail_types) & analytics_events["regime"].isin(detail_regimes)
            ].copy()
            event_columns = [
                "signal_date", "symbol", "name", "signal_type", "regime", "signal_score", "entry_date",
                "entry_price", "shares", "return_1d_pct", "return_3d_pct", "return_5d_pct", "return_10d_pct",
                "mfe_10d_pct", "mae_10d_pct", "fillable", "overlaps_prior_10d",
            ]
            event_show = filtered_events[event_columns].sort_values("signal_date", ascending=False).head(1000).rename(
                columns={
                    "signal_date": "信号日", "symbol": "代码", "name": "名称", "signal_type": "信号",
                    "regime": "市场状态", "signal_score": "评分", "entry_date": "模拟买入日",
                    "entry_price": "模拟买入价", "shares": "股数", "return_1d_pct": "1日%",
                    "return_3d_pct": "3日%", "return_5d_pct": "5日%", "return_10d_pct": "10日%",
                    "mfe_10d_pct": "MFE%", "mae_10d_pct": "MAE%", "fillable": "可成交",
                    "overlaps_prior_10d": "与前信号重叠",
                }
            )
            st.dataframe(event_show, hide_index=True, width="stretch", height=560)
            download_a, download_b = st.columns(2)
            with download_a:
                st.download_button(
                    "下载全部事件CSV",
                    analytics_files["events"].read_bytes(),
                    file_name="双高火箭事件明细.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with download_b:
                st.download_button(
                    "下载完整统计报告",
                    analytics_files["report"].read_bytes(),
                    file_name="普通双高与火箭统计报告.md",
                    mime="text/markdown",
                    width="stretch",
                )
with turtle_tab:
    st.markdown("#### 逆势海龟研究")
    st.caption(
        "正式海龟的大盘条件：对应板块指数达到跌幅，或上证指数跌幅超过设定值；"
        "同时必须满足个股逆势涨幅和20日量比1.2-2.0。该规则用于历史检验，不等同于经典海龟交易法。"
    )
    rank_action, rank_note = st.columns([1.25, 3])
    with rank_action:
        if st.button("创业板实时涨幅前15 → 加入自选", type="primary", width="stretch"):
            with st.spinner("正在读取创业板实时行情、计算涨幅榜和盘中海龟预候选..."):
                try:
                    growth_rank = fetch_growth_top_rank(config, limit=15)
                    changed = add_rank_to_watchlist(growth_rank.rows, WATCHLIST_PATH)
                    growth_rank.rows.to_csv(GROWTH_TOP15_PATH, index=False, encoding="utf-8-sig")
                except Exception as error:
                    st.error(f"创业板实时榜读取失败：{error}")
                else:
                    previews = int(growth_rank.rows["turtle_preview"].sum())
                    st.session_state["market_sync_flash"] = (
                        f"创业板实时涨幅前15已加入系统自选（新增或更新{changed}只）；"
                        f"盘中海龟预候选{previews}只。收盘后仍须更新完整日线确认。"
                    )
                    st.cache_data.clear()
                    st.rerun()
    with rank_note:
        st.warning(
            "问财选股页被同花顺资讯窗口遮挡时，系统使用通达信实时行情兜底。"
            "盘中量比仍会变化，预候选不是正式买入信号；15:10后更新行情并重算海龟。"
        )

    if GROWTH_TOP15_PATH.exists():
        growth_top15 = pd.read_csv(GROWTH_TOP15_PATH, dtype={"symbol": str}, encoding="utf-8-sig")
        growth_top15["symbol"] = growth_top15["symbol"].astype(str).str.zfill(6)
        rank_columns = [
            "rank", "symbol", "name", "gain_pct", "price", "volume_ratio20_preview",
            "turtle_preview", "market_trigger", "growth_index_return_pct", "shanghai_return_pct",
            "quote_time", "source",
        ]
        rank_show = growth_top15[[column for column in rank_columns if column in growth_top15.columns]].copy()
        for column in ("gain_pct", "volume_ratio20_preview", "growth_index_return_pct", "shanghai_return_pct"):
            if column in rank_show.columns:
                rank_show[column] = pd.to_numeric(rank_show[column], errors="coerce").round(2)
        st.dataframe(
            rank_show.rename(
                columns={
                    "rank": "排名", "symbol": "代码", "name": "名称", "gain_pct": "实时涨幅%",
                    "price": "现价", "volume_ratio20_preview": "盘中量比20",
                    "turtle_preview": "海龟预候选", "market_trigger": "大盘触发",
                    "growth_index_return_pct": "创业板指%", "shanghai_return_pct": "上证指数%",
                    "quote_time": "行情时间", "source": "数据来源",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    turtle_rule_a, turtle_rule_b, turtle_rule_c, turtle_rule_d, turtle_rule_e, turtle_rule_f, turtle_rule_g = st.columns(
        [0.9, 0.9, 0.95, 1.05, 0.8, 0.8, 1.2]
    )
    with turtle_rule_a:
        turtle_index_drop = st.number_input(
            "板块指数跌幅 %", min_value=0.5, max_value=10.0,
            value=abs(float(config.get("turtle_index_drop_pct", -2.0))), step=0.5,
        )
    with turtle_rule_b:
        turtle_shanghai_drop = st.number_input(
            "上证跌幅 %", min_value=0.5, max_value=10.0,
            value=abs(float(config.get("turtle_shanghai_drop_pct", -1.0))), step=0.5,
        )
    with turtle_rule_c:
        turtle_ordinary_gain = st.number_input(
            "沪深主板上涨至少 %", min_value=1.0, max_value=20.0,
            value=float(config.get("turtle_ordinary_gain_pct", 5.0)), step=0.5,
        )
    with turtle_rule_d:
        turtle_growth_gain = st.number_input(
            "创业板/科创板上涨至少 %", min_value=1.0, max_value=30.0,
            value=float(config.get("turtle_growth_gain_pct", 10.0)), step=0.5,
        )
    with turtle_rule_e:
        turtle_min_volume = st.number_input(
            "最小量比20", min_value=0.1, max_value=10.0,
            value=float(config.get("turtle_min_volume_ratio", 1.2)), step=0.1,
        )
    with turtle_rule_f:
        turtle_max_volume = st.number_input(
            "最大量比20", min_value=0.1, max_value=20.0,
            value=float(config.get("turtle_max_volume_ratio", 2.0)), step=0.1,
        )
    with turtle_rule_g:
        st.write("")
        if st.button("更新指数并重算海龟", type="primary", width="stretch"):
            turtle_progress = st.progress(0.0, text="正在更新上证、深证、创业板和科创50...")
            try:
                update_benchmark_prices(
                    BENCHMARK_PATH,
                    BENCHMARK_REPORT,
                    days=800,
                    progress=lambda done, total, message: turtle_progress.progress(
                        min(0.08, done / max(1, total) * 0.08), text=message
                    ),
                )
                saved_config = dict(config)
                saved_config.update(
                    {
                        "turtle_index_drop_pct": -abs(float(turtle_index_drop)),
                        "turtle_shanghai_drop_pct": -abs(float(turtle_shanghai_drop)),
                        "turtle_ordinary_gain_pct": float(turtle_ordinary_gain),
                        "turtle_growth_gain_pct": float(turtle_growth_gain),
                        "turtle_min_volume_ratio": float(turtle_min_volume),
                        "turtle_max_volume_ratio": float(turtle_max_volume),
                    }
                )
                CONFIG_PATH.write_text(json.dumps(saved_config, ensure_ascii=False, indent=2), encoding="utf-8")
                current_benchmarks = load_benchmarks(str(BENCHMARK_PATH), BENCHMARK_PATH.stat().st_mtime)
                turtle_result = analyze_turtle_events(
                    prices,
                    current_benchmarks,
                    saved_config,
                    names=dict(zip(watchlist["code"], watchlist["name"])),
                    progress=lambda done, total, message: turtle_progress.progress(
                        0.08 + min(0.92, done / max(1, total) * 0.92), text=message
                    ),
                )
                save_turtle_bundle(turtle_result, TURTLE_ANALYTICS_DIR)
                turtle_reversal_result = analyze_turtle_reversal(prices, turtle_result.events, saved_config)
                save_turtle_reversal_bundle(turtle_reversal_result, TURTLE_REVERSAL_DIR)
            except Exception as error:
                turtle_progress.empty()
                st.error(f"海龟研究生成失败：{error}")
            else:
                turtle_progress.progress(1.0, text=f"完成，共识别{len(turtle_result.events)}条海龟事件")
                st.cache_data.clear()
                st.rerun()

    turtle_files = {
        "events": TURTLE_ANALYTICS_DIR / "events_latest.csv",
        "summary": TURTLE_ANALYTICS_DIR / "summary_latest.csv",
        "board": TURTLE_ANALYTICS_DIR / "board_summary_latest.csv",
        "validation": TURTLE_ANALYTICS_DIR / "validation_latest.csv",
        "walkforward": TURTLE_ANALYTICS_DIR / "walkforward_latest.csv",
        "stability": TURTLE_ANALYTICS_DIR / "stability_latest.csv",
        "metadata": TURTLE_ANALYTICS_DIR / "metadata_latest.json",
        "report": TURTLE_ANALYTICS_DIR / "report_latest.md",
    }
    turtle_ready = all(path.exists() for path in turtle_files.values())
    if not turtle_ready:
        st.info("尚未生成海龟报告。点击上方按钮联网更新指数并计算。")
    else:
        turtle_metadata = json.loads(turtle_files["metadata"].read_text(encoding="utf-8"))
        turtle_summary = load_analytics_csv(str(turtle_files["summary"]), turtle_files["summary"].stat().st_mtime)
        turtle_board = load_analytics_csv(str(turtle_files["board"]), turtle_files["board"].stat().st_mtime)
        turtle_validation = load_analytics_csv(str(turtle_files["validation"]), turtle_files["validation"].stat().st_mtime)
        turtle_walkforward = load_analytics_csv(str(turtle_files["walkforward"]), turtle_files["walkforward"].stat().st_mtime)
        turtle_stability = load_analytics_csv(str(turtle_files["stability"]), turtle_files["stability"].stat().st_mtime)
        turtle_event_rows = load_analytics_csv(str(turtle_files["events"]), turtle_files["events"].stat().st_mtime)
        current_price_end = prices["date"].max().strftime("%Y-%m-%d")
        if turtle_metadata.get("price_end") != current_price_end:
            st.warning(f"海龟报告行情截至 {turtle_metadata.get('price_end', '--')}，当前行情已到 {current_price_end}，请重新计算。")

        turtle_10 = turtle_summary.loc[turtle_summary["horizon_days"] == 10] if not turtle_summary.empty else pd.DataFrame()
        turtle_oos = turtle_validation.loc[turtle_validation["sample"] == "样本外40%"] if not turtle_validation.empty else pd.DataFrame()
        turtle_assessment = turtle_stability.iloc[0]["assessment"] if not turtle_stability.empty else "样本不足"
        latest_turtle_date = turtle_event_rows["signal_date"].max() if not turtle_event_rows.empty else "--"
        turtle_metrics = st.columns(6)
        turtle_metrics[0].metric("海龟事件", int(turtle_metadata.get("events", 0)))
        turtle_metrics[1].metric("最近事件日", latest_turtle_date)
        turtle_metrics[2].metric("10日有效样本", int(turtle_metadata.get("usable_10d_samples", 0)))
        turtle_metrics[3].metric("10日平均收益", f"{float(turtle_10.iloc[0]['avg_return_pct']):.2f}%" if not turtle_10.empty else "--")
        turtle_metrics[4].metric("样本外10日均值", f"{float(turtle_oos.iloc[0]['avg_return_pct']):.2f}%" if not turtle_oos.empty else "--")
        turtle_metrics[5].metric("稳定性", turtle_assessment)
        st.caption(
            f"研究区间 {turtle_metadata.get('research_start')} 至 {turtle_metadata.get('price_end')} · "
            f"指数行情至 {turtle_metadata.get('benchmark_end')} · 统一模拟市值 {float(turtle_metadata.get('analysis_trade_value', 0)):,.2f}元"
        )

        turtle_overview, turtle_validate, turtle_detail = st.tabs(["效果统计", "样本外与稳定性", "事件明细"])
        with turtle_overview:
            if not turtle_summary.empty:
                turtle_chart = go.Figure()
                turtle_chart.add_bar(
                    name="个股净收益",
                    x=[f"{int(value)}日" for value in turtle_summary["horizon_days"]],
                    y=turtle_summary["avg_return_pct"],
                    marker_color="#0b7285",
                    text=[f"{value:.2f}%" for value in turtle_summary["avg_return_pct"]],
                    textposition="outside",
                )
                turtle_chart.add_bar(
                    name="相对指数超额",
                    x=[f"{int(value)}日" for value in turtle_summary["horizon_days"]],
                    y=turtle_summary["avg_excess_return_pct"],
                    marker_color="#f08c00",
                    text=[f"{value:.2f}%" for value in turtle_summary["avg_excess_return_pct"]],
                    textposition="outside",
                )
                turtle_chart.update_layout(
                    barmode="group", height=360, margin=dict(l=20, r=20, t=30, b=20),
                    yaxis_title="收益率 %", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(turtle_chart, width="stretch", config={"displaylogo": False})
                st.dataframe(turtle_summary, hide_index=True, width="stretch")
            if not turtle_board.empty:
                st.markdown("**板块与基准分组（10日）**")
                st.dataframe(turtle_board, hide_index=True, width="stretch")
        with turtle_validate:
            if turtle_validation.empty:
                st.info("当前有效日期不足，暂不能切分样本内外。")
            else:
                st.dataframe(turtle_validation, hide_index=True, width="stretch")
            if not turtle_walkforward.empty:
                st.markdown("**扩展窗口滚动验证**")
                st.dataframe(turtle_walkforward, hide_index=True, width="stretch")
            st.markdown("**稳定性结论**")
            st.dataframe(turtle_stability, hide_index=True, width="stretch")
            st.warning("只有样本外和多数滚动窗口仍保持正向，才值得继续观察；历史统计不保证下一次事件获利。")
        with turtle_detail:
            if turtle_event_rows.empty:
                st.info("当前阈值下没有识别到海龟事件。")
            else:
                turtle_symbols = turtle_event_rows[["symbol", "name"]].copy()
                turtle_symbols["symbol"] = turtle_symbols["symbol"].astype(str).str.zfill(6)
                turtle_symbols = turtle_symbols.drop_duplicates("symbol")
                turtle_symbols["label"] = turtle_symbols.apply(
                    lambda row: f"{row['symbol']}  {row['name'] if row['name'] and row['name'] != row['symbol'] else ''}".strip(),
                    axis=1,
                )
                turtle_label_to_code = dict(zip(turtle_symbols["label"], turtle_symbols["symbol"]))
                turtle_chart_a, turtle_chart_b = st.columns([3, 1])
                with turtle_chart_a:
                    turtle_selected_label = st.selectbox(
                        "海龟事件股票",
                        list(turtle_label_to_code),
                        key="turtle_chart_symbol",
                    )
                with turtle_chart_b:
                    turtle_chart_period = st.segmented_control(
                        "海龟K线区间",
                        options=[f"{display_year}年", "120日", "250日"],
                        default=f"{display_year}年",
                        key="turtle_chart_period",
                    )
                turtle_code = turtle_label_to_code[turtle_selected_label]
                turtle_name = turtle_symbols.loc[turtle_symbols["symbol"] == turtle_code, "name"].iloc[0]
                turtle_stock = enrich_indicators(prices.loc[prices["symbol"] == turtle_code])
                turtle_dates_for_stock = set(
                    pd.to_datetime(
                        turtle_event_rows.loc[
                            turtle_event_rows["symbol"].astype(str).str.zfill(6) == turtle_code,
                            "signal_date",
                        ],
                        errors="coerce",
                    )
                )
                st.markdown("**🐢**＝逆势涨幅达标且20日量比在设定范围内；本图不显示双高菱形、火箭和双高卖出标志。")
                st.plotly_chart(
                    make_chart(
                        turtle_stock,
                        f"{turtle_code} {turtle_name} · 海龟策略独立复核",
                        turtle_chart_period or f"{display_year}年",
                        turtle_dates_for_stock,
                        strategy="turtle",
                    ),
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": True},
                )
                turtle_columns = [column for column in [
                    "signal_date", "symbol", "name", "board", "market_trigger", "benchmark_name",
                    "volume_ratio20", "stock_return_pct", "benchmark_return_pct", "shanghai_return_pct",
                    "relative_strength_pct", "entry_date", "return_1d_pct",
                    "return_3d_pct", "return_5d_pct", "return_10d_pct", "excess_return_10d_pct",
                    "fillable", "overlaps_prior_10d",
                ] if column in turtle_event_rows.columns]
                event_show = turtle_event_rows[turtle_columns].sort_values("signal_date", ascending=False).rename(
                    columns={
                        "signal_date": "事件日", "symbol": "代码", "name": "名称", "board": "板块",
                        "market_trigger": "大盘触发", "benchmark_name": "对比指数",
                        "volume_ratio20": "当日量比20", "stock_return_pct": "个股当日%",
                        "benchmark_return_pct": "板块指数当日%", "shanghai_return_pct": "上证当日%",
                        "relative_strength_pct": "逆势强度%",
                        "entry_date": "模拟买入日", "return_1d_pct": "1日%", "return_3d_pct": "3日%",
                        "return_5d_pct": "5日%", "return_10d_pct": "10日%",
                        "excess_return_10d_pct": "10日超额%", "fillable": "可成交",
                        "overlaps_prior_10d": "与前事件重叠",
                    }
                )
                st.dataframe(event_show, hide_index=True, width="stretch", height=560)
            turtle_download_a, turtle_download_b = st.columns(2)
            with turtle_download_a:
                st.download_button(
                    "下载海龟事件CSV", turtle_files["events"].read_bytes(),
                    file_name="逆势海龟事件明细.csv", mime="text/csv", width="stretch",
                )
            with turtle_download_b:
                st.download_button(
                    "下载海龟研究报告", turtle_files["report"].read_bytes(),
                    file_name="逆势海龟研究报告.md", mime="text/markdown", width="stretch",
                )

    reversal_files = {
        "events": TURTLE_REVERSAL_DIR / "events_latest.csv",
        "summary": TURTLE_REVERSAL_DIR / "summary_latest.csv",
        "metadata": TURTLE_REVERSAL_DIR / "metadata_latest.json",
        "report": TURTLE_REVERSAL_DIR / "report_latest.md",
    }
    with st.expander("片仔癀型低位反转海龟（独立研究观察）", expanded=True):
        if not all(path.exists() for path in reversal_files.values()):
            st.info("尚未生成独立观察报告。点击上方“更新指数并重算海龟”后自动生成。")
        else:
            reversal_metadata = json.loads(reversal_files["metadata"].read_text(encoding="utf-8"))
            reversal_summary = load_analytics_csv(
                str(reversal_files["summary"]), reversal_files["summary"].stat().st_mtime
            )
            reversal_events = load_analytics_csv(
                str(reversal_files["events"]), reversal_files["events"].stat().st_mtime
            )
            a_oos = reversal_summary.loc[
                (reversal_summary["cohort"] == "A级五项全满足")
                & (reversal_summary["sample"] == "样本外2026")
                & (reversal_summary["exit_method"] == "5%移动保护")
            ]
            reversal_metrics = st.columns(5)
            reversal_metrics[0].metric("A级累计", int(reversal_metadata.get("grade_a_events", 0)))
            reversal_metrics[1].metric("B级累计", int(reversal_metadata.get("grade_b_events", 0)))
            reversal_metrics[2].metric("A级样本外", int(reversal_metadata.get("grade_a_oos_usable", 0)))
            reversal_metrics[3].metric(
                "样本外5%保护均值",
                f"{float(a_oos.iloc[0]['avg_return_pct']):.2f}%" if not a_oos.empty else "--",
            )
            reversal_metrics[4].metric("正式准入", reversal_metadata.get("promotion_gate", "未通过"))
            st.warning(
                "A级必须同时满足温和量能、一阳收复MA5/MA10/MA20、突破10日平台、"
                "仍在下降MA60下方、强势收盘。它不自动进入模拟账户，也不替代原海龟或双高。"
            )
            if not reversal_events.empty:
                reversal_show = reversal_events.loc[
                    reversal_events["研究等级"].astype(str).ne("")
                ].copy()
                reversal_columns = [column for column in [
                    "signal_date", "symbol", "name", "研究等级", "反转评分", "stock_return_pct",
                    "volume_ratio20", "relative_strength_pct", "return_10d_pct",
                    "trailing_5_return_pct", "trailing_8_return_pct", "sample",
                ] if column in reversal_show.columns]
                st.dataframe(
                    reversal_show[reversal_columns].sort_values("signal_date", ascending=False),
                    hide_index=True,
                    width="stretch",
                    height=360,
                )
            st.caption(reversal_metadata.get("promotion_rule", ""))
            st.download_button(
                "下载片仔癀型研究报告",
                reversal_files["report"].read_bytes(),
                file_name="片仔癀型低位反转海龟研究.md",
                mime="text/markdown",
                width="stretch",
            )
# 历史均线/突破研究源码仅作内部追溯，禁止作为裸字符串交给 Streamlit 渲染。
_LEGACY_RESEARCH_SOURCE = '''with ma_tab:
    st.warning("均线研究已暂停：当前不会重算，也不作为活动候选03的买入依据；历史报告仅保留查看。")
    st.markdown("#### 均线买点稳定性")
    st.caption("独立检验MA5首上MA20、K线首上MA5、震荡不破MA20后再上MA5；不覆盖双高和海龟策略。")
    ma_control_a, ma_control_b, ma_control_c = st.columns([1, 1, 1.5])
    with ma_control_a:
        ma_consolidation_days = st.number_input(
            "震荡观察日数", min_value=3, max_value=20,
            value=int(config.get("ma_consolidation_days", 5)), step=1,
        )
    with ma_control_b:
        ma_consolidation_range = st.number_input(
            "震荡区间上限 %", min_value=1.0, max_value=20.0,
            value=float(config.get("ma_consolidation_range_pct", 8.0)), step=0.5,
        )
    with ma_control_c:
        st.write("")
        if LEGACY_RESEARCH_ENABLED and st.button("重算均线稳定性", type="primary", width="stretch"):
            ma_progress = st.progress(0.0, text="正在分析三类均线买点...")
            try:
                saved_config = dict(config)
                saved_config.update(
                    {
                        "ma_consolidation_days": int(ma_consolidation_days),
                        "ma_consolidation_range_pct": float(ma_consolidation_range),
                    }
                )
                CONFIG_PATH.write_text(json.dumps(saved_config, ensure_ascii=False, indent=2), encoding="utf-8")
                ma_result = analyze_ma_strategy(
                    prices,
                    saved_config,
                    names=dict(zip(watchlist["code"], watchlist["name"])),
                    progress=lambda done, total, message: ma_progress.progress(
                        min(0.85, done / max(1, total) * 0.85), text=message
                    ),
                )
                save_ma_strategy_bundle(ma_result, MA_STRATEGY_DIR)
                ma_progress.progress(0.88, text="事件统计完成，正在运行四组5%-8%账户回测...")
                run_ma_portfolio_variants(load_prices_csv(PRICE_PATH), saved_config, MA_STRATEGY_DIR)
            except Exception as error:
                ma_progress.empty()
                st.error(f"均线稳定性研究失败：{error}")
            else:
                ma_progress.progress(1.0, text="均线稳定性与组合回测完成")
                st.cache_data.clear()
                st.rerun()

    ma_files = {
        "summary": MA_STRATEGY_DIR / "summary_latest.csv",
        "validation": MA_STRATEGY_DIR / "validation_latest.csv",
        "stability": MA_STRATEGY_DIR / "stability_latest.csv",
        "portfolio": MA_STRATEGY_DIR / "portfolio_summary_latest.csv",
        "metadata": MA_STRATEGY_DIR / "metadata_latest.json",
        "report": MA_STRATEGY_DIR / "report_latest.md",
    }
    ma_ready = all(path.exists() for path in ma_files.values())
    if not ma_ready:
        st.info("尚未生成均线稳定性报告。点击上方按钮计算。")
    else:
        ma_metadata = json.loads(ma_files["metadata"].read_text(encoding="utf-8"))
        ma_summary = load_analytics_csv(str(ma_files["summary"]), ma_files["summary"].stat().st_mtime)
        ma_validation = load_analytics_csv(str(ma_files["validation"]), ma_files["validation"].stat().st_mtime)
        ma_stability = load_analytics_csv(str(ma_files["stability"]), ma_files["stability"].stat().st_mtime)
        ma_portfolio = load_analytics_csv(str(ma_files["portfolio"]), ma_files["portfolio"].stat().st_mtime)
        current_price_end = prices["date"].max().strftime("%Y-%m-%d")
        if ma_metadata.get("price_end") != current_price_end:
            st.warning(f"均线报告行情截至 {ma_metadata.get('price_end', '--')}，当前行情已到 {current_price_end}，请重新计算。")

        ma_combined_10 = ma_summary.loc[
            (ma_summary["strategy"] == "三类买点合并") & (ma_summary["horizon_days"] == 10)
        ]
        ma_combined_oos = ma_validation.loc[
            (ma_validation["strategy"] == "三类买点合并") & (ma_validation["sample"] == "样本外40%")
        ]
        ma_combined_portfolio = ma_portfolio.loc[ma_portfolio["variant"] == "combined"]
        ma_metrics = st.columns(6)
        ma_metrics[0].metric("研究事件", int(ma_metadata.get("events", 0)))
        ma_metrics[1].metric("合并10日均值", f"{float(ma_combined_10.iloc[0]['avg_return_pct']):.2f}%" if not ma_combined_10.empty else "--")
        ma_metrics[2].metric("合并样本外", f"{float(ma_combined_oos.iloc[0]['avg_return_pct']):.2f}%" if not ma_combined_oos.empty else "--")
        ma_metrics[3].metric("组合收益", f"{float(ma_combined_portfolio.iloc[0]['total_return_pct']):.2f}%" if not ma_combined_portfolio.empty else "--")
        ma_metrics[4].metric("组合最大回撤", f"{float(ma_combined_portfolio.iloc[0]['max_drawdown_pct']):.2f}%" if not ma_combined_portfolio.empty else "--")
        ma_metrics[5].metric("最终判断", "不稳定" if not ma_combined_portfolio.empty and bool(ma_combined_portfolio.iloc[0]["halted"]) else "继续观察")
        if not ma_combined_portfolio.empty and bool(ma_combined_portfolio.iloc[0]["halted"]):
            st.warning("固定10日事件均值为正，但真实组合触发8%硬熔断。重叠信号不能当作独立盈利样本，本策略当前不能进入模拟买入。")

        ma_effect_tab, ma_validation_tab, ma_portfolio_tab = st.tabs(["固定周期效果", "样本外稳定性", "5%-8%组合回测"])
        with ma_effect_tab:
            ma_ten_day = ma_summary.loc[ma_summary["horizon_days"] == 10].copy()
            st.dataframe(
                ma_ten_day[
                    [
                        "strategy", "count", "win_rate_pct", "avg_return_pct", "median_return_pct",
                        "avg_gain_pct", "avg_loss_pct", "payoff_ratio", "avg_mfe_pct", "avg_mae_pct",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption("固定10日统计允许不同股票和同一股票的信号重叠，因此只用于判断信号方向，不代表账户可实现收益。")
        with ma_validation_tab:
            ma_oos = ma_validation.loc[ma_validation["sample"] == "样本外40%"]
            st.dataframe(ma_oos, hide_index=True, width="stretch")
            st.markdown("**重叠与稳定性**")
            st.dataframe(ma_stability, hide_index=True, width="stretch")
        with ma_portfolio_tab:
            portfolio_columns = [
                "strategy", "total_return_pct", "annual_return_pct", "max_drawdown_pct", "trade_count",
                "win_rate_pct", "sharpe_ratio", "profit_factor", "halted", "halt_date",
            ]
            st.dataframe(ma_portfolio[portfolio_columns], hide_index=True, width="stretch")
            st.caption("组合口径：2024-01-01起交易，最多5仓、单仓18%，收盘确认次日开盘执行，5%暂停开仓、8%清仓停机。")
            st.download_button(
                "下载均线稳定性完整报告",
                ma_files["report"].read_bytes(),
                file_name="均线买点稳定性研究报告.md",
                mime="text/markdown",
                width="stretch",
            )
with breakout_tab:
    st.warning("突破研究已暂停：主升突破分支已从活动候选03中关闭；历史报告仅保留查看。")
    st.markdown("#### A股强趋势突破研究")
    breakout_head_a, breakout_head_b = st.columns([3, 1])
    with breakout_head_a:
        st.caption("多头趋势、10日缩量箱体、放量中大阳首次突破20/60日新高；排除ST及日均成交额不足5000万元标的。")
    with breakout_head_b:
        if LEGACY_RESEARCH_ENABLED and st.button("重算突破稳定性", type="primary", width="stretch"):
            breakout_progress = st.progress(0.0, text="正在检验A股强趋势突破事件...")
            try:
                breakout_result = analyze_breakout_strategy(
                    prices,
                    config,
                    names=dict(zip(watchlist["code"], watchlist["name"])),
                    progress=lambda done, total, message: breakout_progress.progress(
                        min(0.82, done / max(1, total) * 0.82), text=message
                    ),
                )
                save_breakout_bundle(breakout_result, BREAKOUT_STRATEGY_DIR)
                excluded_symbols = watchlist.loc[
                    watchlist["name"].astype(str).str.upper().str.contains("ST|退", regex=True), "code"
                ].astype(str).str.zfill(6).tolist()
                breakout_progress.progress(0.86, text="事件统计完成，正在运行20日和60日组合回测...")
                run_breakout_portfolios(
                    load_prices_csv(PRICE_PATH),
                    config,
                    BREAKOUT_STRATEGY_DIR,
                    excluded_symbols,
                )
            except Exception as error:
                breakout_progress.empty()
                st.error(f"突破策略研究失败：{error}")
            else:
                breakout_progress.progress(1.0, text="突破策略稳定性研究完成")
                st.cache_data.clear()
                st.rerun()

    breakout_files = {
        "summary": BREAKOUT_STRATEGY_DIR / "summary_latest.csv",
        "validation": BREAKOUT_STRATEGY_DIR / "validation_latest.csv",
        "stability": BREAKOUT_STRATEGY_DIR / "stability_latest.csv",
        "portfolio": BREAKOUT_STRATEGY_DIR / "portfolio_summary_latest.csv",
        "metadata": BREAKOUT_STRATEGY_DIR / "metadata_latest.json",
        "report": BREAKOUT_STRATEGY_DIR / "report_latest.md",
    }
    breakout_ready = all(path.exists() for path in breakout_files.values())
    if not breakout_ready:
        st.info("尚未生成突破策略报告。点击上方按钮计算。")
    else:
        breakout_metadata = json.loads(breakout_files["metadata"].read_text(encoding="utf-8"))
        breakout_summary = load_analytics_csv(str(breakout_files["summary"]), breakout_files["summary"].stat().st_mtime)
        breakout_validation = load_analytics_csv(str(breakout_files["validation"]), breakout_files["validation"].stat().st_mtime)
        breakout_stability = load_analytics_csv(str(breakout_files["stability"]), breakout_files["stability"].stat().st_mtime)
        breakout_portfolio = load_analytics_csv(str(breakout_files["portfolio"]), breakout_files["portfolio"].stat().st_mtime)
        current_price_end = prices["date"].max().strftime("%Y-%m-%d")
        if breakout_metadata.get("price_end") != current_price_end:
            st.warning(f"突破报告行情截至 {breakout_metadata.get('price_end', '--')}，当前行情已到 {current_price_end}，请重新计算。")
        breakout_60_10 = breakout_summary.loc[
            (breakout_summary["strategy"] == "60日强趋势突破") & (breakout_summary["horizon_days"] == 10)
        ]
        breakout_60_oos = breakout_validation.loc[
            (breakout_validation["strategy"] == "60日强趋势突破") & (breakout_validation["sample"] == "样本外40%")
        ]
        breakout_60_portfolio = breakout_portfolio.loc[breakout_portfolio["window"] == 60]
        breakout_metrics = st.columns(6)
        breakout_metrics[0].metric("突破事件", int(breakout_metadata.get("events", 0)))
        breakout_metrics[1].metric("60日10日均值", f"{float(breakout_60_10.iloc[0]['avg_return_pct']):.2f}%" if not breakout_60_10.empty else "--")
        breakout_metrics[2].metric("60日样本外", f"{float(breakout_60_oos.iloc[0]['avg_return_pct']):.2f}%" if not breakout_60_oos.empty else "--")
        breakout_metrics[3].metric("组合收益", f"{float(breakout_60_portfolio.iloc[0]['total_return_pct']):.2f}%" if not breakout_60_portfolio.empty else "--")
        breakout_metrics[4].metric("组合最大回撤", f"{float(breakout_60_portfolio.iloc[0]['max_drawdown_pct']):.2f}%" if not breakout_60_portfolio.empty else "--")
        breakout_metrics[5].metric("最终判断", "不可用" if not breakout_60_portfolio.empty and bool(breakout_60_portfolio.iloc[0]["halted"]) else "继续观察")
        if not breakout_60_portfolio.empty and bool(breakout_60_portfolio.iloc[0]["halted"]):
            st.warning("20日和60日版本均触发8%硬熔断。当前严格规则不能作为“见到就买”方案，也不会进入模拟买入。")

        breakout_effect_tab, breakout_validation_tab, breakout_portfolio_tab = st.tabs(["事件效果", "样本外验证", "5%-8%组合回测"])
        with breakout_effect_tab:
            st.dataframe(
                breakout_summary.loc[breakout_summary["horizon_days"] == 10],
                hide_index=True,
                width="stretch",
            )
        with breakout_validation_tab:
            st.dataframe(
                breakout_validation.loc[breakout_validation["sample"] == "样本外40%"],
                hide_index=True,
                width="stretch",
            )
            st.markdown("**稳定性**")
            st.dataframe(breakout_stability, hide_index=True, width="stretch")
            st.caption("60日版本样本外仅21条，当前数据量不足以证明长期稳定。")
        with breakout_portfolio_tab:
            portfolio_columns = [
                "strategy", "total_return_pct", "annual_return_pct", "max_drawdown_pct", "trade_count",
                "win_rate_pct", "sharpe_ratio", "profit_factor", "halted", "halt_date",
            ]
            st.dataframe(breakout_portfolio[portfolio_columns], hide_index=True, width="stretch")
            st.caption("A股适配：2024年起交易，T+1次日执行，主板6%单票熔断、创业/科创8%，账户5%暂停、8%清仓。")
            st.download_button(
                "下载A股强趋势突破报告",
                breakout_files["report"].read_bytes(),
                file_name="A股强趋势突破稳定性报告.md",
                mime="text/markdown",
                width="stretch",
            )
'''
del _LEGACY_RESEARCH_SOURCE
with taozi_tab:
    st.markdown("#### 质量趋势精选")
    st.caption(
        "原“偷桃子”研究已改名为“质量趋势精选”：先看质量与估值，再看中长期趋势，"
        "回撤只用于排序和风控，不把单纯低价当作买入理由。"
    )
    st.warning("这里提供候选筛选和人工复核，不是自动下单。系统推荐必须先更新行情和财务数据。")

    taozi_report = REPORT_DIR / "taozi_research.md"
    taozi_script = REPORT_DIR / "taozi_research.py"
    taozi_run_a, taozi_run_b = st.columns([1, 3])
    with taozi_run_a:
        if st.button("重算质量趋势精选", type="primary", width="stretch"):
            if not taozi_script.exists():
                st.error(f"研究脚本不存在：{taozi_script}")
            else:
                runner = MARKET_DATA_PYTHON if MARKET_DATA_PYTHON.exists() else Path("python")
                with st.spinner("正在用本地点时数据重算质量趋势精选..."):
                    try:
                        result = subprocess.run(
                            [str(runner), str(taozi_script)],
                            cwd=str(REPORT_DIR),
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=600,
                        )
                        if result.returncode != 0:
                            st.error("重算失败，请查看下方输出。")
                            if result.stderr:
                                st.code(result.stderr[-4000:], language="text")
                        else:
                            st.success("质量趋势精选报告已更新。")
                            st.cache_data.clear()
                            st.rerun()
                    except Exception as error:
                        st.error(f"重算异常：{error}")
    with taozi_run_b:
        st.caption("研究报告文件：reports/taozi_research.md")

    if taozi_report.exists():
        taozi_text = taozi_report.read_text(encoding="utf-8", errors="replace")
        with st.expander("查看质量趋势精选研究结论", expanded=True):
            st.markdown(taozi_text)
        st.download_button(
            "下载质量趋势精选研究报告",
            taozi_text,
            file_name="质量趋势精选研究报告.md",
            mime="text/markdown",
            width="stretch",
        )
    else:
        st.info("尚未生成质量趋势精选报告，请点击“重算质量趋势精选”。")

    st.markdown("#### 价值成长回调 · 研究版")
    st.caption(
        "营收同比>20%、净利率>15%、PE 0~30，趋势向上且收盘回到MA20附近；"
        "信号日收盘后次日开盘买入，持有5个完整交易日。严格按A股T+1模拟，暂不作为正式推荐。"
    )
    value_run_col, value_file_col = st.columns([1, 3])
    with value_run_col:
        if st.button("回测价值成长回调", type="secondary", width="stretch"):
            if not VALUE_GROWTH_SCRIPT.exists():
                st.error(f"研究脚本不存在：{VALUE_GROWTH_SCRIPT}")
            else:
                runner = MARKET_DATA_PYTHON if MARKET_DATA_PYTHON.exists() else Path("python")
                with st.spinner("正在按公告日和T+1口径回测价值成长回调..."):
                    try:
                        result = subprocess.run(
                            [str(runner), str(VALUE_GROWTH_SCRIPT)],
                            cwd=str(REPORT_DIR),
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=900,
                        )
                        if result.returncode != 0:
                            st.error("价值成长回调回测失败，请查看输出。")
                            if result.stderr:
                                st.code(result.stderr[-5000:], language="text")
                        else:
                            st.success("价值成长回调研究报告已更新。")
                            st.cache_data.clear()
                            st.rerun()
                    except Exception as error:
                        st.error(f"价值成长回测异常：{error}")
    with value_file_col:
        st.caption("研究报告：reports/value_growth_pullback_backtest/value_growth_pullback_report.md")

    if VALUE_GROWTH_REPORT.exists():
        value_text = VALUE_GROWTH_REPORT.read_text(encoding="utf-8", errors="replace")
        st.warning("研究版结果未达到正式策略门槛：仅供候选观察和模拟验证，不自动下单。")
        with st.expander("查看价值成长回调回测结论", expanded=False):
            st.markdown(value_text)
        st.download_button(
            "下载价值成长回调研究报告",
            value_text,
            file_name="价值成长回调研究报告.md",
            mime="text/markdown",
            width="stretch",
        )
        if VALUE_GROWTH_CANDIDATES.exists() and VALUE_GROWTH_CANDIDATES.stat().st_size:
            value_candidates = pd.read_csv(VALUE_GROWTH_CANDIDATES, dtype={"ts_code": str})
            if not value_candidates.empty:
                value_display = value_candidates.rename(
                    columns={
                        "signal_date": "信号日",
                        "ts_code": "代码",
                        "current_name": "名称",
                        "close": "收盘",
                        "ma20": "MA20",
                        "ma20_distance_pct": "距MA20%",
                        "pullback_pct": "距前20日高点%",
                        "pe": "PE",
                        "or_yoy": "营收同比",
                        "netprofit_margin": "净利率",
                        "entry_date": "预计买入日",
                    }
                )
                keep = [c for c in ["信号日", "代码", "名称", "收盘", "MA20", "距MA20%", "距前20日高点%", "PE", "营收同比", "净利率", "预计买入日"] if c in value_display]
                st.dataframe(value_display[keep], hide_index=True, width="stretch")
    else:
        st.info("尚未生成价值成长回调研究报告，请点击“回测价值成长回调”。")

    st.markdown("#### 我的选股 · 系统判断")
    st.caption(
        "输入代码后，系统会结合最新本地行情判断趋势、双高、量价和回撤状态；"
        "这一步是辅助复核，不把技术信号冒充成基本面结论。"
    )
    my_pick_text = st.text_input(
        "我的选股代码",
        placeholder="例如：000598, 300483, 603259",
        key="quality_my_pick_text",
    )
    judge_col, add_col = st.columns([1, 1])
    with judge_col:
        judge_my_picks = st.button("系统判断我的选股", type="primary", width="stretch")
    with add_col:
        add_my_picks = st.button("加入系统自选并扫描", width="stretch")

    def _parse_codes(raw: str) -> list[str]:
        values = re.split(r"[\s,，;；]+", raw or "")
        return list(dict.fromkeys(v.strip().zfill(6) for v in values if v.strip()))

    my_codes = _parse_codes(my_pick_text)
    if my_codes:
        invalid = [code for code in my_codes if not SUPPORTED_STOCK_CODE.fullmatch(code)]
        if invalid:
            st.error(f"代码格式不正确：{', '.join(invalid)}")
        valid_codes = [code for code in my_codes if SUPPORTED_STOCK_CODE.fullmatch(code)]
        if add_my_picks and valid_codes:
            added = []
            for code in valid_codes:
                item, was_added = add_watch_item(
                    WATCHLIST_PATH,
                    code,
                    "",
                    tags="质量趋势精选 我的选股 待扫描",
                    source="质量趋势精选",
                )
                added.append(f"{item.code}{'（新增）' if was_added else '（已存在）'}")
            st.session_state["quality_pick_flash"] = "、".join(added)
            st.cache_data.clear()
            st.rerun()

        if judge_my_picks:
            judgement_rows = []
            for code in valid_codes:
                stock = prices.loc[prices["symbol"].astype(str).str.zfill(6) == code].copy()
                if stock.empty:
                    judgement_rows.append(
                        {
                            "代码": code,
                            "行情状态": "无本地行情",
                            "系统判断": "先更新行情",
                            "理由": "本地行情中没有该代码",
                        }
                    )
                    continue
                calculated = calculate_signals(stock)
                last = calculated.iloc[-1]
                if bool(last.get("buy_candidate", False)):
                    verdict = "高质量买入候选"
                elif bool(last.get("raw_double_high", False)):
                    verdict = "普通双高，观察"
                elif bool(last.get("trend_ok", False)):
                    verdict = "趋势通过，等待突破"
                else:
                    verdict = "暂不介入"
                reasons = []
                if not bool(last.get("trend_ok", False)):
                    reasons.append("趋势未通过")
                if not bool(last.get("first_breakout", False)):
                    reasons.append("非首次突破")
                if not bool(last.get("volume_ok", False)):
                    reasons.append("量比不足或过热")
                if not bool(last.get("candle_ok", False)):
                    reasons.append("K线质量不足")
                if not bool(last.get("extension_ok", False)):
                    reasons.append("偏离MA20过大")
                judgement_rows.append(
                    {
                        "代码": code,
                        "行情日期": pd.Timestamp(last["date"]).strftime("%Y-%m-%d"),
                        "收盘": round(float(last["close"]), 2),
                        "信号评分": int(last.get("signal_score", 0)),
                        "系统判断": verdict,
                        "理由": "；".join(reasons) if reasons else "趋势、突破、量价和K线条件均通过",
                    }
                )
            st.dataframe(pd.DataFrame(judgement_rows), hide_index=True, width="stretch")

    quality_flash = st.session_state.pop("quality_pick_flash", "")
    if quality_flash:
        st.success(f"已处理：{quality_flash}")

    st.markdown("#### 系统自动推荐")
    if QUALITY_CANDIDATES_PATH.exists() and QUALITY_CANDIDATES_PATH.stat().st_size:
        recommended = pd.read_csv(QUALITY_CANDIDATES_PATH, dtype={"ts_code": str})
        if not recommended.empty:
            latest_recommendation_date = str(recommended["trade_date"].max())[:10]
            st.caption(
                f"系统推荐来自最近一个可完整验证持有期的月末：{latest_recommendation_date}。"
                "行情更新到今天后，可在“我的选股”输入代码做当前技术复核。"
            )
            display = recommended.rename(
                columns={
                    "ts_code": "代码",
                    "trade_date": "信号日",
                    "close": "收盘",
                    "roe": "ROE",
                    "netprofit_yoy": "净利润同比",
                    "pe": "PE",
                    "drawdown_from_120d_high": "距120日高点回撤",
                    "score": "综合分",
                }
            )
            keep = [c for c in ["信号日", "代码", "收盘", "ROE", "净利润同比", "PE", "距120日高点回撤", "综合分"] if c in display]
            st.dataframe(display[keep], hide_index=True, width="stretch")
            recommend_codes = [str(code).split(".")[0].zfill(6) for code in recommended["ts_code"].dropna()]
            if st.button("把系统推荐全部加入自选", key="quality_add_recommended", width="stretch"):
                for code in recommend_codes:
                    add_watch_item(
                        WATCHLIST_PATH,
                        code,
                        "",
                        tags="质量趋势精选 系统推荐 待扫描",
                        source="质量趋势精选",
                    )
                st.success(f"已将 {len(recommend_codes)} 只系统推荐加入自选。请更新行情后再扫描。")
                st.cache_data.clear()
        else:
            st.info("当前没有通过质量趋势精选的系统推荐。")
    else:
        st.info("还没有系统推荐结果，请先点击“重算质量趋势精选”。")

with pullback_tab:
    st.markdown("#### 实验性回调起爆 · 独立模拟策略")
    pullback_head_a, pullback_head_b = st.columns([3, 1])
    with pullback_head_a:
        st.caption(
            "与双高、海龟、均线完全分开。强趋势中等待10日缩量回调，再由放量大阳突破平台触发；"
            "仅接本地模拟账户，不连接券商。"
        )
    with pullback_head_b:
        if st.button("重扫回调起爆", type="primary", width="stretch"):
            with st.spinner("正在扫描自选池中的回调起爆信号..."):
                scan_pullback_signals(prices, watchlist, PULLBACK_SCAN_PATH, config, recent_days=800)
            st.cache_data.clear()
            st.rerun()

    st.warning(
        "当前回测属于小样本试运行：样本内8笔、样本外7笔；样本外收益为正但不足以证明长期稳定。"
        "系统允许模拟验证，不允许实盘自动下单。"
    )
    pullback_metrics = st.columns(6)
    pullback_metrics[0].metric("样本内交易", 8)
    pullback_metrics[1].metric("样本内收益", "+7.70%")
    pullback_metrics[2].metric("样本外交易", 7)
    pullback_metrics[3].metric("样本外收益", "+6.14%")
    pullback_metrics[4].metric("样本外最大回撤", "2.26%")
    pullback_metrics[5].metric("当前级别", "实验模拟")

    rules = pullback_rules_from_config(config)
    st.caption(
        f"固定参数：60日涨幅≥{rules.rise60_threshold_pct:g}% · 回调{rules.pullback_min_pct:g}%–"
        f"{rules.pullback_max_pct:g}% · 回调量≤前段{rules.pullback_volume_ratio:.0%} · "
        f"触发量比≥{rules.min_volume_ratio:g} · 主板止损{rules.main_stop_pct:g}% · "
        f"创业/科创止损{rules.growth_stop_pct:g}% · 最高收盘回撤{rules.trailing_stop_pct:g}%清仓。"
    )

    if pullback_scan.empty:
        st.info("当前历史行情中尚无回调起爆扫描结果。点击“重扫回调起爆”。")
    else:
        scan_show = pullback_scan.copy()
        scan_show["date"] = scan_show["date"].dt.strftime("%Y-%m-%d")
        show_columns = [
            "date", "symbol", "name", "board", "daily_gain_pct", "volume_ratio20",
            "rise60_pct", "pullback_pct", "box_range_pct", "initial_stop_pct", "trailing_stop_pct",
        ]
        st.dataframe(
            scan_show[show_columns].rename(
                columns={
                    "date": "信号日", "symbol": "代码", "name": "名称", "board": "板块",
                    "daily_gain_pct": "当日涨幅%", "volume_ratio20": "20日量比",
                    "rise60_pct": "60日区间涨幅%", "pullback_pct": "回调幅度%",
                    "box_range_pct": "平台振幅%", "initial_stop_pct": "初始止损%",
                    "trailing_stop_pct": "移动熔断%",
                }
            ),
            hide_index=True,
            width="stretch",
        )

        signal_labels = [
            f"{row.date:%Y-%m-%d}  {row.symbol}  {row.name}"
            for row in pullback_scan.itertuples()
        ]
        selected_signal_label = st.selectbox("查看回调起爆信号", signal_labels, key="pullback_signal_select")
        selected_signal = pullback_scan.iloc[signal_labels.index(selected_signal_label)]
        pullback_code = str(selected_signal["symbol"]).zfill(6)
        pullback_name = str(selected_signal["name"])
        pullback_stock = enrich_indicators(prices.loc[prices["symbol"] == pullback_code])
        pullback_stock = calculate_pullback_signals(
            pullback_stock,
            is_growth_board(pullback_code),
            rules,
        )
        st.plotly_chart(
            make_chart(
                pullback_stock,
                f"{pullback_code} {pullback_name} · 回调起爆独立K线",
                chart_period or f"{display_year}年",
                strategy=PULLBACK_STRATEGY,
            ),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )

        current_paper = load_paper_account(PAPER_PATHS) or paper_state
        latest_stock_date = pullback_stock.iloc[-1]["date"].strftime("%Y-%m-%d")
        selected_signal_date = pd.Timestamp(selected_signal["date"]).strftime("%Y-%m-%d")
        latest_signal_valid = bool(pullback_stock.iloc[-1]["pullback_ignition"])
        is_current_signal = latest_signal_valid and selected_signal_date == latest_stock_date
        account_ready = current_paper["last_processed_date"] == latest_stock_date
        account_equity = float(current_paper["equity_history"][-1]["equity"])
        maximum_value = min(
            float(current_paper["cash"]),
            account_equity * float(config["max_position_pct"]) / 100.0,
        )
        order_a, order_b, order_c = st.columns([1.4, 1.8, 1])
        with order_a:
            pullback_target = st.number_input(
                "回调起爆计划市值（元）",
                min_value=100.0,
                max_value=max(100.0, maximum_value),
                value=max(100.0, min(10000.0, maximum_value)),
                step=1000.0,
                key=f"pullback_target_{pullback_code}",
            )
        with order_b:
            pullback_confirmed = st.checkbox(
                "确认仅为模拟委托，不连接券商",
                key=f"pullback_confirm_{pullback_code}",
            )
            if not is_current_signal:
                st.caption("所选为历史信号，只能查看，禁止事后补单。")
            elif not account_ready:
                st.caption("请先到“模拟账户”把账户推进到最新行情日。")
        with order_c:
            st.write("")
            can_queue_pullback = (
                bool(config.get("pullback_ignition_enabled", True))
                and is_current_signal
                and account_ready
                and pullback_confirmed
                and current_paper["risk_lock"]["status"] == RISK_NORMAL
                and pullback_code not in current_paper["positions"]
                and pullback_code not in current_paper["pending_entries"]
            )
            if st.button(
                "提交起爆模拟买入",
                type="primary",
                width="stretch",
                disabled=not can_queue_pullback,
            ):
                if not any(item["symbol"] == pullback_code for item in current_paper.get("paper_watchlist", [])):
                    add_paper_watch_symbol(PAPER_PATHS, pullback_code, pullback_name)
                pullback_rows = [
                    Candle(
                        row.date.strftime("%Y-%m-%d"), pullback_code, float(row.open), float(row.high),
                        float(row.low), float(row.close), float(row.volume),
                    )
                    for row in pullback_stock.itertuples()
                ]
                try:
                    queue_paper_buy_signal(
                        PAPER_PATHS,
                        {pullback_code: pullback_rows},
                        config,
                        pullback_code,
                        float(pullback_target),
                        pullback_confirmed,
                        strategy=PULLBACK_STRATEGY,
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success("回调起爆模拟买单已排队，下一可交易日开盘按滑点和整手规则成交。")
                    st.rerun()
with lifecycle_tab:
    st.markdown("#### 女鬼生命周期95% · 独立模拟策略")
    st.caption(
        "女人第1日后的第二个交易日满足95%价格接近度和1.8量比才介入；"
        "10%盈利减半、3%高点回撤减半、5%回撤清仓，女神上轨首次突破可加5%。"
    )
    st.info(
        "图标口径：红色“女”星＝女鬼周期介入候选；空心橙三角＝女神阶段上轨观察，"
        "只允许已有持仓加仓，不是首买。没有红色“女”星时，不按女鬼策略新开仓；"
        "生命周期离场由模拟账户结合实际成本、最高价回撤和MA5判断。"
    )
    lifecycle_head_a, lifecycle_head_b = st.columns([3, 1])
    with lifecycle_head_a:
        st.warning(
            "当前95%版历史样本仍少，2024-2026只有5笔交易；本页仅用于模拟观察，不连接券商。"
        )
    with lifecycle_head_b:
        if st.button("重扫女鬼生命周期", type="primary", width="stretch"):
            with st.spinner("正在扫描女鬼→女人→女神信号..."):
                scan_lifecycle_signals(prices, watchlist, LIFECYCLE_SCAN_PATH, config, recent_days=800)
            st.cache_data.clear()
            st.rerun()

    life_rules = lifecycle_rules_from_config(config)
    current_lifecycle_count = (
        int((pd.to_numeric(lifecycle_scan.get("age_trade_days"), errors="coerce") == 0).sum())
        if not lifecycle_scan.empty and "age_trade_days" in lifecycle_scan.columns
        else 0
    )
    life_metrics = st.columns(5)
    life_metrics[0].metric("今日候选", current_lifecycle_count, f"历史通过 {len(lifecycle_scan)}")
    life_metrics[1].metric("价格接近度", f"{life_rules.price_proximity_pct:g}%")
    life_metrics[2].metric("最低量比", f"{life_rules.min_volume_ratio:g}")
    life_metrics[3].metric("盈利减半", f"{config.get('ghost_lifecycle_profit_reduce_pct', 10):g}%")
    life_metrics[4].metric("回撤清仓", f"{config.get('ghost_lifecycle_drawdown_exit_pct', 5):g}%")

    if lifecycle_scan.empty:
        st.info("当前行情文件没有女鬼生命周期候选。点击“重扫女鬼生命周期”更新。")
    else:
        if current_lifecycle_count == 0:
            st.warning(
                "最新完整交易日没有正式介入候选。下表是近800个交易日的历史通过记录，"
                "用于复盘，不代表今天可以买入。"
            )
        life_show = lifecycle_scan.copy()
        life_show["date"] = pd.to_datetime(life_show["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(
            life_show[
                ["date", "symbol", "name", "close", "pct_chg", "volume_ratio",
                 "price_proximity_pct", "ma120", "boll_mid", "boll_upper"]
            ].rename(
                columns={
                    "date": "信号日", "symbol": "代码", "name": "名称", "close": "收盘",
                    "pct_chg": "涨幅%", "volume_ratio": "量比", "price_proximity_pct": "距60日高点%",
                    "ma120": "MA120", "boll_mid": "BOLL中轨", "boll_upper": "BOLL上轨",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        life_labels = [
            f"{row.date:%Y-%m-%d}  {row.symbol}  {row.name}"
            for row in lifecycle_scan.itertuples()
        ]
        selected_life_label = st.selectbox("查看女鬼周期信号", life_labels, key="lifecycle_signal_select")
        selected_life = lifecycle_scan.iloc[life_labels.index(selected_life_label)]
        life_code = str(selected_life["symbol"]).zfill(6)
        life_name = str(selected_life["name"])
        life_stock = calculate_lifecycle_signals(
            prices.loc[prices["symbol"] == life_code], life_rules
        )
        st.plotly_chart(
            make_chart(
                life_stock,
                f"{life_code} {life_name} · 女鬼生命周期95%",
                chart_period or f"{display_year}年",
                strategy=LIFECYCLE_STRATEGY,
            ),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )
        current_paper = load_paper_account(PAPER_PATHS) or paper_state
        latest_life_date = life_stock.iloc[-1]["date"].strftime("%Y-%m-%d")
        selected_life_date = pd.Timestamp(selected_life["date"]).strftime("%Y-%m-%d")
        is_current_life = bool(life_stock.iloc[-1]["eligible"]) and selected_life_date == latest_life_date
        account_ready = current_paper["last_processed_date"] == latest_life_date
        account_equity = float(current_paper["equity_history"][-1]["equity"])
        initial_pct = 5.0
        maximum_value = min(float(current_paper["cash"]), account_equity * initial_pct / 100.0)
        order_a, order_b, order_c = st.columns([1.4, 1.8, 1])
        with order_a:
            life_target = st.number_input(
                "女鬼周期计划市值（元）",
                min_value=100.0,
                max_value=max(100.0, maximum_value),
                value=max(100.0, min(5000.0, maximum_value)),
                step=1000.0,
                key=f"lifecycle_target_{life_code}",
            )
            st.caption(f"初始仓位上限：{maximum_value:,.2f}元；女神上轨最多再加5%")
        with order_b:
            life_confirmed = st.checkbox(
                "确认仅为模拟委托，不连接券商",
                key=f"lifecycle_confirm_{life_code}",
            )
            if not is_current_life:
                st.caption("所选为历史信号，只能查看，禁止事后补单。")
            elif not account_ready:
                st.caption("请先到“模拟账户”推进到账户最新行情日。")
        with order_c:
            st.write("")
            can_queue_life = (
                bool(config.get("ghost_lifecycle_enabled", True))
                and is_current_life
                and account_ready
                and life_confirmed
                and current_paper["risk_lock"]["status"] == RISK_NORMAL
                and life_code not in current_paper["positions"]
                and life_code not in current_paper["pending_entries"]
            )
            if st.button(
                "提交女鬼周期模拟买入",
                type="primary",
                width="stretch",
                disabled=not can_queue_life,
            ):
                if not any(item["symbol"] == life_code for item in current_paper.get("paper_watchlist", [])):
                    add_paper_watch_symbol(PAPER_PATHS, life_code, life_name)
                life_rows = [
                    Candle(
                        row.date.strftime("%Y-%m-%d"), life_code, float(row.open), float(row.high),
                        float(row.low), float(row.close), float(row.volume),
                    )
                    for row in life_stock.itertuples()
                ]
                try:
                    queue_paper_buy_signal(
                        PAPER_PATHS,
                        {life_code: life_rows},
                        config,
                        life_code,
                        float(life_target),
                        life_confirmed,
                        strategy=LIFECYCLE_STRATEGY,
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success("女鬼生命周期模拟买单已排队，下一可交易日开盘按T+1规则成交。")
                    st.rerun()
with candidate03_tab:
    st.markdown("#### 候选03观察 · 收盘后手工交易信号")
    st.error(
        "候选03当前启用：首板强势 + 近5日涨幅≤12%。研究样本仍有限，"
        "这里只做观察和模拟验证，不保证盈利，不连接同花顺或任何券商。"
    )
    st.caption(
        "买入信号在每个已完成交易日收盘后生成；卖出条件每日收盘检查。"
        "你在同花顺成交后，必须把实际成本和股数登记到下方，系统才有依据生成逐票卖出提醒。"
    )

    update_col, scan_col, rule_col = st.columns([1.2, 1, 2.2])
    with update_col:
        if st.button("Tushare收盘更新并扫描", type="primary", width="stretch", key="candidate03_refresh"):
            with st.spinner("正在增量更新全市场行情、财务快照并重算候选03，可能需要数分钟..."):
                ok, detail = run_candidate03("refresh")
            if ok:
                st.cache_data.clear()
                st.session_state["candidate03_flash"] = detail
                st.rerun()
            st.error(detail)
    with scan_col:
        if st.button("仅重扫本地数据库", width="stretch", key="candidate03_scan"):
            with st.spinner("正在使用本地数据库重算候选03..."):
                ok, detail = run_candidate03("scan")
            if ok:
                st.cache_data.clear()
                st.session_state["candidate03_flash"] = "本地候选03扫描已完成。"
                st.rerun()
            st.error(detail)
    with rule_col:
        st.info("执行规则：收盘确认信号，下一可交易日开盘由你手工操作；单票参考上限5%，总仓位40%。")
    if "candidate03_flash" in st.session_state:
        st.success(st.session_state.pop("candidate03_flash"))

    with st.expander("同花顺问财全自动对话（试验）", expanded=False):
        st.warning(
            "仅控制 hexin.exe 左侧问财面板，明确排除 xiadan.exe；会自动输入、发送并读取回答，"
            "不会点击行情表或任何交易按钮。运行时请保持同花顺已登录且问财面板可见。"
        )
        iwencai_question = st.text_area(
            "要问问财的问题",
            value=(
                "请只回答三点：在A股T+1制度下，低波动质量价值轮动策略以沪深300跌破60日均线"
                "作为停止开仓条件，最大的三个风险是什么？"
            ),
            max_chars=500,
            key="iwencai_question",
        )
        if st.button("全自动询问问财", type="primary", width="stretch", key="ask_iwencai"):
            if len(" ".join(iwencai_question.split())) < 2:
                st.error("请输入至少2个字符的问题。")
                st.stop()
            with st.spinner("正在控制同花顺左侧问财面板并等待回答，请暂时不要操作鼠标键盘..."):
                ok, detail = run_iwencai(iwencai_question)
            if ok:
                st.session_state["candidate03_flash"] = "问财已回答，结果已读取。"
                st.rerun()
            st.error(detail)
        iwencai_response = load_iwencai_response()
        if iwencai_response:
            answer = re.sub(r"[\ue000-\uf8ff]", "", str(iwencai_response.get("excerpt", "")))
            st.caption(f"最近询问：{iwencai_response.get('asked_at', '--')}")
            st.text_area("问财最近回答", answer, height=260, disabled=True)

    candidate_status = load_candidate03_status()
    if not candidate_status:
        st.info("还没有候选03扫描结果，请先点击“仅重扫本地数据库”。")
    else:
        status_cols = st.columns(6)
        status_cols[0].metric("行情日", candidate_status.get("data_date", "--"))
        status_cols[1].metric("决策日", candidate_status.get("decision_date", "--"))
        status_cols[2].metric("当日新买点", "是" if candidate_status.get("new_week_signal") else "否")
        status_cols[3].metric("市场闸门", "开放" if candidate_status.get("market_gate_active") else "关闭")
        status_cols[4].metric("手工卖出提醒", int(candidate_status.get("sell_signals", 0)))
        status_cols[5].metric("财务公告到", candidate_status.get("financial_data_through") or "--")
        if candidate_status.get("mirror_rule_active"):
            blocked = int(candidate_status.get("mirror_blocked_candidates", 0))
            escaped = int(candidate_status.get("mirror_escape_signals", 0))
            st.info(
                "照镜子覆盖层已启用：默认观察 MA5、MA20/BOLL中轨。"
                f"本次候选禁买 {blocked} 只，持仓逃跑提醒 {escaped} 只；"
                "跌破关键线后反抽仍站不回，只能作为逃跑/禁买信号，不会生成买点。"
            )
        csi_close = candidate_status.get("csi300_close")
        csi_ma60 = candidate_status.get("csi300_ma60")
        if csi_close is not None and csi_ma60 is not None:
            risk_text = "风险退出" if candidate_status.get("market_risk_off") else "风险正常"
            market_data_date = candidate_status.get("market_data_date", "--")
            st.caption(
                f"沪深300：数据截至 {market_data_date}，收盘 {float(csi_close):.2f} / "
                f"MA60 {float(csi_ma60):.2f} · {risk_text}"
            )

    candidate_buys = (
        load_analytics_csv(str(CANDIDATE03_BUY_PATH), CANDIDATE03_BUY_PATH.stat().st_mtime)
        if CANDIDATE03_BUY_PATH.exists() else pd.DataFrame()
    )
    candidate_last = (
        load_analytics_csv(str(CANDIDATE03_LAST_WEEK_PATH), CANDIDATE03_LAST_WEEK_PATH.stat().st_mtime)
        if CANDIDATE03_LAST_WEEK_PATH.exists() else pd.DataFrame()
    )
    candidate_sells = (
        load_analytics_csv(str(CANDIDATE03_SELL_PATH), CANDIDATE03_SELL_PATH.stat().st_mtime)
        if CANDIDATE03_SELL_PATH.exists() else pd.DataFrame()
    )

    signal_view, holding_view = st.tabs(["买卖清单", "同花顺实际持仓登记"])
    with signal_view:
        st.markdown("##### 当日新买入清单")
        if candidate_buys.empty:
            st.info("当前没有新买入信号。周中不会重复发上周买点，市场闸门关闭时也不会选股。")
        else:
            buy_columns = [
                column for column in [
                    "trade_date", "symbol", "name", "rank", "score", "close_qfq",
                    "mirror_state", "mirror_reason", "execution",
                ]
                if column in candidate_buys.columns
            ]
            st.dataframe(candidate_buys[buy_columns], hide_index=True, width="stretch")
            st.download_button(
                "下载当日买入清单CSV", CANDIDATE03_BUY_PATH.read_bytes(),
                file_name=f"候选03买入清单_{candidate_status.get('data_date', 'latest')}.csv",
                mime="text/csv", width="stretch",
            )

        st.markdown("##### 当日候选参考")
        if candidate_last.empty:
            st.caption("当前交易日未通过市场闸门，或没有满足条件的前8名。")
        else:
            last_columns = [
                column for column in [
                    "trade_date", "symbol", "name", "rank", "score", "close_qfq",
                    "mirror_state", "mirror_reason",
                ]
                if column in candidate_last.columns
            ]
            st.dataframe(candidate_last[last_columns], hide_index=True, width="stretch")
            if not candidate_status.get("new_week_signal"):
                st.warning("这是历史交易日参考，不是今天的新买入指令，禁止事后补买。")

        st.markdown("##### 已登记持仓的卖出检查")
        if candidate_sells.empty:
            st.info("尚未登记同花顺实际持仓，因此没有逐票卖出检查。")
        else:
            sell_columns = [
                column for column in [
                    "signal_date", "symbol", "name", "entry_price", "latest_close", "pnl_pct",
                    "holding_sessions", "weekly_rank", "action", "reason",
                    "mirror_state", "mirror_reason", "execution",
                ] if column in candidate_sells.columns
            ]
            st.dataframe(candidate_sells[sell_columns], hide_index=True, width="stretch")
            st.download_button(
                "下载持仓卖出检查CSV", CANDIDATE03_SELL_PATH.read_bytes(),
                file_name=f"候选03卖出检查_{candidate_status.get('data_date', 'latest')}.csv",
                mime="text/csv", width="stretch",
            )

    with holding_view:
        st.warning("这里只登记你已经在同花顺实际成交的持仓；填写表单不会买卖股票。卖出成交后请手工移除。")
        with st.form("candidate03_holding_form", clear_on_submit=True):
            holding_a, holding_b, holding_c = st.columns(3)
            with holding_a:
                holding_symbol = st.text_input("股票代码", placeholder="例如 600000")
                holding_name = st.text_input("股票名称", placeholder="例如 浦发银行")
            with holding_b:
                holding_date = st.date_input("实际买入日期")
                holding_price = st.number_input("实际成本价", min_value=0.001, value=10.0, step=0.01)
            with holding_c:
                holding_shares = st.number_input("实际持股数", min_value=100, value=100, step=100)
                holding_notes = st.text_input("备注", placeholder="例如 同花顺手工成交")
                holding_add_watch = st.checkbox(
                    "新代码同时加入系统总自选",
                    value=True,
                    help="加入后，双高、海龟、回调起爆和女鬼周期扫描都会覆盖这只股票。",
                )
            add_holding = st.form_submit_button("登记实际持仓", type="primary", width="stretch")
        if add_holding:
            try:
                clean_holding_symbol = str(holding_symbol).strip()
                if not SUPPORTED_STOCK_CODE.fullmatch(clean_holding_symbol):
                    raise ValueError("请输入受支持的6位A股代码")
                holding_watch_added = False
                if clean_holding_symbol not in watch_codes:
                    if not holding_add_watch:
                        raise ValueError(
                            f"{clean_holding_symbol} 尚未加入系统自选。请勾选“新代码同时加入系统总自选”后再登记。"
                        )
                    _, holding_watch_added = add_watch_item(
                        WATCHLIST_PATH,
                        clean_holding_symbol,
                        str(holding_name).strip(),
                        tags="候选03持仓 手工新增 待更新行情",
                        source="候选03持仓登记",
                    )
                add_candidate03_holding(
                    CANDIDATE03_HOLDINGS_PATH, clean_holding_symbol, holding_name, holding_date,
                    float(holding_price), int(holding_shares), holding_notes,
                )
                ok, detail = run_candidate03("scan")
                if not ok:
                    raise RuntimeError(detail)
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                st.session_state["candidate03_flash"] = (
                    "实际持仓已登记，代码已同步加入系统自选，并完成卖出条件重算。请联网更新补齐行情。"
                    if holding_watch_added
                    else "实际持仓已登记并完成卖出条件重算。"
                )
                st.cache_data.clear()
                st.rerun()

        candidate_holdings = load_candidate03_holdings(CANDIDATE03_HOLDINGS_PATH)
        if candidate_holdings.empty:
            st.caption("当前没有已登记的候选03实际持仓。")
        else:
            st.dataframe(candidate_holdings, hide_index=True, width="stretch")
            remove_labels = [f"{row.symbol} {row.name}" for row in candidate_holdings.itertuples()]
            remove_col, remove_button_col = st.columns([3, 1])
            with remove_col:
                remove_label = st.selectbox("卖出成交后选择要移除的持仓", remove_labels)
            with remove_button_col:
                st.write("")
                if st.button("确认已卖出并移除", width="stretch"):
                    remove_candidate03_holding(CANDIDATE03_HOLDINGS_PATH, remove_label.split()[0])
                    run_candidate03("scan")
                    st.cache_data.clear()
                    st.rerun()
with risk_tab:
    if signals.empty:
        st.info("暂无风控事件")
    else:
        risk_actions = ["PAUSE_BUY", "RESUME_BUY", "HALT", "SELL_BLOCKED", "BUY_CANCELLED"]
        risk_events = signals.loc[signals["action"].isin(risk_actions), ["date", "action", "drawdown_pct", "reason"]].tail(30)
        st.dataframe(risk_events, hide_index=True, width="stretch")

with paper_tab:
    paper_state = load_paper_account(PAPER_PATHS) or paper_state
    paper_latest = paper_state["equity_history"][-1]
    risk_status = paper_state["risk_lock"]["status"]
    risk_labels = PAPER_RISK_LABELS

    paper_head_a, paper_head_b, paper_head_c = st.columns([2.4, 1, 1.2])
    with paper_head_a:
        st.markdown("#### 持久化模拟账户")
        st.caption(
            f"账户 {paper_state['account_id']} · 开户日 {paper_state['opening_date']} · "
            f"已处理至 {paper_state['last_processed_date']}"
        )
    with paper_head_b:
        if st.button("推进至最新行情", type="primary", width="stretch"):
            with st.spinner("正在按交易日推进模拟账户并生成日报..."):
                price_map = load_prices_csv(PRICE_PATH)
                advance_paper_account(PAPER_PATHS, price_map, config)
            st.rerun()
    with paper_head_c:
        latest_report = PAPER_PATHS.reports / "latest.md"
        if latest_report.exists():
            st.download_button(
                "下载最新日报",
                data=latest_report.read_bytes(),
                file_name=f"模拟账户日报_{paper_state['last_processed_date']}.md",
                mime="text/markdown",
                width="stretch",
            )

    newest_market_date = prices["date"].max().strftime("%Y-%m-%d")
    if paper_state["last_processed_date"] < newest_market_date:
        st.info(f"发现新行情：{newest_market_date}。点击“推进至最新行情”后，按日期逐日成交并自动生成日报。")
    elif risk_status == RISK_HARD_LOCK:
        st.error(risk_labels[risk_status])
    elif risk_status == RISK_SOFT_LOCK:
        st.warning(risk_labels[risk_status])
    else:
        st.success(risk_labels[risk_status])

    paper_metrics = st.columns(6)
    paper_metrics[0].metric("模拟权益", f"{float(paper_latest['equity']):,.2f}")
    paper_metrics[1].metric("可用现金", f"{float(paper_latest['cash']):,.2f}")
    paper_metrics[2].metric("持仓市值", f"{float(paper_latest['market_value']):,.2f}")
    paper_metrics[3].metric("当日盈亏", f"{float(paper_latest['daily_pnl']):+,.2f}")
    paper_metrics[4].metric("当前回撤", f"{float(paper_latest['drawdown_pct']):.2f}%")
    paper_metrics[5].metric("风险状态", risk_labels[risk_status].split(" · ")[0])

    if risk_status == RISK_SOFT_LOCK:
        st.markdown("##### 人工解除5%暂停")
        unlock_a, unlock_b, unlock_c = st.columns([2.4, 1.5, 1])
        with unlock_a:
            unlock_reason = st.text_input(
                "复核说明",
                placeholder="例如：已核对持仓、未成交委托和最新行情，确认继续模拟",
                key="paper_unlock_reason",
            )
        with unlock_b:
            unlock_confirmed = st.checkbox("我已完成人工复核", key="paper_unlock_confirmed")
        with unlock_c:
            st.write("")
            if st.button("解除暂停", width="stretch"):
                try:
                    unlock_soft_risk(PAPER_PATHS, unlock_reason, unlock_confirmed)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.rerun()

    paper_view_a, paper_view_b, paper_view_c = st.tabs(["持仓与净值", "成交与事件", "账户管理"])
    with paper_view_a:
        positions = pd.DataFrame(list(paper_state["positions"].values()))
        if positions.empty:
            st.info("当前模拟账户无持仓。新买入信号在收盘确认，下一可交易日开盘模拟成交。")
        else:
            positions["strategy"] = positions.get("strategy", "quality_double_high")
            positions["latest_price"] = positions["symbol"].map(paper_state.get("last_prices", {}))
            positions["market_value"] = positions["shares"] * positions["latest_price"]
            positions["pnl_pct"] = (positions["latest_price"] / positions["entry_price"] - 1) * 100
            show_positions = positions[["symbol", "strategy", "shares", "entry_price", "entry_date", "latest_price", "market_value", "pnl_pct"]].copy()
            show_positions.columns = ["代码", "策略", "数量", "成本价", "建仓日", "最新价", "市值", "浮动盈亏%"]
            st.dataframe(show_positions, hide_index=True, width="stretch")
        paper_equity = pd.DataFrame(paper_state["equity_history"])
        if not paper_equity.empty:
            paper_equity["date"] = pd.to_datetime(paper_equity["date"])
            st.line_chart(paper_equity.set_index("date")[["equity"]], height=280)
            st.dataframe(
                paper_equity.tail(20).sort_values("date", ascending=False).rename(
                    columns={
                        "date": "日期", "equity": "权益", "cash": "现金", "market_value": "持仓市值",
                        "positions": "持仓数", "daily_pnl": "当日盈亏", "daily_return_pct": "日收益%",
                        "drawdown_pct": "回撤%", "risk_status": "风险状态",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
    with paper_view_b:
        paper_trades = pd.DataFrame(paper_state["trades"])
        if paper_trades.empty:
            st.info("暂无已平仓交易。")
        else:
            st.markdown("**已平仓交易**")
            st.dataframe(paper_trades.tail(50).sort_values("exit_date", ascending=False), hide_index=True, width="stretch")
        paper_events = pd.DataFrame(paper_state["events"])
        st.markdown("**账户事件**")
        st.dataframe(paper_events.tail(100).iloc[::-1], hide_index=True, width="stretch")
    with paper_view_c:
        st.warning("重置会建立一个新模拟账户，并覆盖当前账户状态。旧的日报文件仍保留在本机。")
        reset_a, reset_b, reset_c = st.columns([1.3, 1.7, 1])
        with reset_a:
            replay_date = st.date_input(
                "新账户起始日",
                value=prices["date"].max().date(),
                min_value=prices["date"].min().date(),
                max_value=prices["date"].max().date(),
                key="paper_replay_date",
            )
        with reset_b:
            reset_confirmed = st.checkbox("确认覆盖当前模拟账户", key="paper_reset_confirmed")
            replay_to_latest = st.checkbox("建立后回放到最新行情日", value=False, key="paper_replay_latest")
        with reset_c:
            st.write("")
            if st.button("重置模拟账户", width="stretch", disabled=not reset_confirmed):
                new_state = initialize_paper_account(
                    PAPER_PATHS,
                    config,
                    replay_date.strftime("%Y-%m-%d"),
                    overwrite=True,
                )
                if replay_to_latest:
                    with st.spinner("正在逐日回放并生成日报..."):
                        new_state = advance_paper_account(PAPER_PATHS, load_prices_csv(PRICE_PATH), config)
                st.rerun()

st.caption(f"净值来源：{equity_path.name if equity_path else '--'} · 3%提前预警 / 5%暂停开仓 / 8%熔断清仓 · 模拟研究，不连接实盘")
