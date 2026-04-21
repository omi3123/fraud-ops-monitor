from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ARTIFACT_DIR, ASSET_DIR, CASE_QUEUE_PATH, DATA_DIR, DEMO_DATA_PATH
from src.inference import load_model


st.set_page_config(
    page_title="Fraud / Anomaly Detection Operations Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOT_TEMPLATE = "plotly_dark"
PRIORITY_ORDER = ["critical", "high", "medium", "low"]
PRIORITY_COLORS = {
    "critical": "#ef4444",
    "high": "#f97316",
    "medium": "#f59e0b",
    "low": "#22c55e",
}


@st.cache_data
def load_demo() -> pd.DataFrame:
    path = DEMO_DATA_PATH
    if not path.exists():
        raise FileNotFoundError("Demo dataset not found. Run scripts/build_demo_assets.py first.")
    return pd.read_csv(path, parse_dates=["event_ts"])


@st.cache_data
def load_alert_queue() -> pd.DataFrame:
    path = CASE_QUEUE_PATH
    if not path.exists():
        raise FileNotFoundError("Alert queue artifact not found. Run scripts/build_demo_assets.py first.")
    return pd.read_csv(path, parse_dates=["event_ts"])


@st.cache_data
def load_hero_metrics() -> dict:
    path = ASSET_DIR / "hero_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def cached_model():
    return load_model(ARTIFACT_DIR)



def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07101c;
            --panel: rgba(10, 18, 32, 0.92);
            --panel-2: rgba(15, 23, 42, 0.92);
            --text: #ecf4ff;
            --muted: #9fb2c9;
            --line: rgba(148, 163, 184, 0.14);
            --shadow: 0 20px 50px rgba(2, 8, 23, 0.42);
            --blue: #38bdf8;
            --indigo: #4f46e5;
            --green: #22c55e;
            --amber: #f59e0b;
            --orange: #f97316;
            --red: #ef4444;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56,189,248,0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(99,102,241,0.12), transparent 32%),
                linear-gradient(180deg, #050b14 0%, #08111f 45%, #091322 100%);
            color: var(--text);
        }
        [data-testid="stHeader"] {background: transparent;}
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7,16,28,0.98), rgba(8,17,31,0.98));
            border-right: 1px solid var(--line);
        }
        section.main > div {padding-top: 1.3rem;}
        .block-container {padding-top: 0.8rem; padding-bottom: 2rem;}
        .hero-shell {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(79,70,229,0.14) 48%, rgba(15,23,42,0.85));
            border: 1px solid rgba(96,165,250,0.22);
            border-radius: 24px;
            padding: 1.45rem 1.55rem 1.35rem 1.55rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -60px -60px auto;
            width: 180px;
            height: 180px;
            background: radial-gradient(circle, rgba(56,189,248,0.24), transparent 70%);
            pointer-events: none;
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            color: #dbeafe;
            background: rgba(96,165,250,0.12);
            border: 1px solid rgba(96,165,250,0.22);
        }
        .hero-title {font-size: 2rem; font-weight: 800; margin: 0.65rem 0 0.2rem 0;}
        .hero-sub {max-width: 880px; color: var(--muted); font-size: 1rem; line-height: 1.55; margin-bottom: 1rem;}
        .badge-row {display:flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 0.35rem;}
        .badge {
            display:inline-flex; align-items:center; gap:0.35rem;
            padding:0.4rem 0.7rem; border-radius:999px; font-size:0.82rem;
            background: rgba(15,23,42,0.55); border:1px solid rgba(148,163,184,0.15); color:#dbe7f5;
        }
        .section-title {font-size: 1.12rem; font-weight: 700; margin: 0.3rem 0 0.9rem 0;}
        .metric-card {
            background: linear-gradient(180deg, rgba(15,23,42,0.86), rgba(10,18,32,0.86));
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            min-height: 132px;
            box-shadow: var(--shadow);
        }
        .metric-label {font-size: 0.82rem; color: var(--muted); margin-bottom: 0.45rem;}
        .metric-value {font-size: 1.9rem; font-weight: 800; line-height: 1.1; margin-bottom: 0.4rem;}
        .metric-sub {font-size: 0.82rem; color: #d6e2f0; opacity: 0.9;}
        .metric-chip {
            display:inline-block; margin-top:0.55rem; padding:0.25rem 0.55rem; border-radius:999px;
            background: rgba(56,189,248,0.12); color:#bfdbfe; font-size:0.74rem; border:1px solid rgba(56,189,248,0.18);
        }
        .panel {
            background: linear-gradient(180deg, rgba(15,23,42,0.78), rgba(10,18,32,0.78));
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 22px;
            padding: 1rem 1rem 0.25rem 1rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }
        .mini-panel {
            background: rgba(15,23,42,0.74); border:1px solid rgba(148,163,184,0.12);
            border-radius:18px; padding:0.9rem 1rem; box-shadow: var(--shadow);
        }
        .sidebar-box {
            background: rgba(15,23,42,0.55);
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 18px;
            padding: 0.9rem;
            margin: 0.2rem 0 1rem 0;
        }
        .stat-grid {
            display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:0.8rem; margin-bottom:0.6rem;
        }
        .sidebar-title {font-weight:700; font-size:1rem; margin-bottom:0.2rem;}
        .sidebar-sub {color: var(--muted); font-size:0.83rem; line-height:1.45;}
        .priority-pill {display:inline-flex; align-items:center; gap:0.35rem; padding:0.28rem 0.6rem; border-radius:999px; font-size:0.76rem;}
        .priority-critical {background: rgba(239,68,68,0.14); border: 1px solid rgba(239,68,68,0.25); color:#fecaca;}
        .priority-high {background: rgba(249,115,22,0.14); border: 1px solid rgba(249,115,22,0.25); color:#fdba74;}
        .priority-medium {background: rgba(245,158,11,0.14); border: 1px solid rgba(245,158,11,0.25); color:#fde68a;}
        .priority-low {background: rgba(34,197,94,0.14); border: 1px solid rgba(34,197,94,0.25); color:#bbf7d0;}
        .table-note {font-size: 0.85rem; color: var(--muted); margin-top: -0.2rem; margin-bottom: 0.8rem;}
        .stDataFrame, .stPlotlyChart, .stFileUploader, .stDownloadButton > button, .stButton > button, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 16px !important;
        }
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #2563eb, #06b6d4) !important;
            color: white !important;
            border: none !important;
            padding: 0.55rem 0.95rem !important;
            font-weight: 600 !important;
            box-shadow: 0 12px 28px rgba(37,99,235,0.28);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: rgba(15,23,42,0.6);
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 16px;
            padding: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto; border-radius: 12px; padding: 0.55rem 0.9rem; color: #c9d8ea;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(6,182,212,0.18));
            color: white !important;
        }
        .small-muted {font-size: 0.82rem; color: var(--muted);}
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_header(filtered: pd.DataFrame) -> None:
    review_queue = int((filtered["needs_review"] == 1).sum())
    critical = int((filtered["priority_band"] == "critical").sum())
    avg_score = filtered["alert_priority_score"].mean()
    st.markdown(
        f"""
        <div class="hero-shell">
            <span class="eyebrow">🛡️ Fraud Ops Command Center</span>
            <div class="hero-title">Fraud / Anomaly Detection Operations Monitor</div>
            <div class="hero-sub">
                Professional command-center style dashboard for transaction anomaly detection, alert prioritization,
                investigator triage, and executive-level operating visibility.
            </div>
            <div class="badge-row">
                <span class="badge">⚡ Review queue: <strong>{review_queue:,}</strong></span>
                <span class="badge">🚨 Critical alerts: <strong>{critical:,}</strong></span>
                <span class="badge">📈 Avg priority score: <strong>{avg_score:,.1f}</strong></span>
                <span class="badge">🏦 Ops-ready workflow demo</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_sidebar(demo: pd.DataFrame):
    st.sidebar.markdown(
        """
        <div class="sidebar-box">
            <div class="sidebar-title">Control Tower</div>
            <div class="sidebar-sub">Filter the live feed, narrow the review queue, and switch between executive, analyst, and governance views.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Workspace",
        options=["Executive Overview", "Alert Queue", "Investigation Workbench", "Model Governance"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div class='small-muted' style='margin:0.25rem 0 0.5rem 0;'>Priority bands</div>", unsafe_allow_html=True)
    bands = st.sidebar.multiselect(
        "Priority band",
        options=PRIORITY_ORDER,
        default=PRIORITY_ORDER,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div class='small-muted' style='margin:0.25rem 0 0.5rem 0;'>Channels</div>", unsafe_allow_html=True)
    channels = st.sidebar.multiselect(
        "Channel",
        options=sorted(demo["channel"].unique().tolist()),
        default=sorted(demo["channel"].unique().tolist()),
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div class='small-muted' style='margin:0.25rem 0 0 0.5rem;'>Merchant categories</div>".replace(' 0 0 0.5rem',' 0 0.5rem 0'), unsafe_allow_html=True)
    categories = st.sidebar.multiselect(
        "Merchant category",
        options=sorted(demo["merchant_category"].unique().tolist()),
        default=sorted(demo["merchant_category"].unique().tolist()),
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div class='small-muted' style='margin:0.25rem 0 0.45rem 0;'>Alert score range</div>", unsafe_allow_html=True)
    score_range = st.sidebar.slider("Alert score range", 0, 100, (0, 100), label_visibility="collapsed")

    selected_count = demo[
        demo["priority_band"].isin(bands)
        & demo["channel"].isin(channels)
        & demo["merchant_category"].isin(categories)
        & demo["alert_priority_score"].between(score_range[0], score_range[1])
    ].shape[0]

    st.sidebar.markdown(
        f"""
        <div class="sidebar-box">
            <div class="sidebar-title">Filtered scope</div>
            <div class="stat-grid">
                <div class="mini-panel"><div class="small-muted">Records</div><div style="font-size:1.15rem;font-weight:700;">{selected_count:,}</div></div>
                <div class="mini-panel"><div class="small-muted">Bands</div><div style="font-size:1.15rem;font-weight:700;">{len(bands)}</div></div>
                <div class="mini-panel"><div class="small-muted">Channels</div><div style="font-size:1.15rem;font-weight:700;">{len(channels)}</div></div>
            </div>
            <div class="sidebar-sub">Use this area during demos to show how operations teams isolate the most relevant alerts quickly.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page, bands, channels, categories, score_range



def filter_demo(demo: pd.DataFrame, bands, channels, categories, score_range) -> pd.DataFrame:
    return demo[
        demo["priority_band"].isin(bands)
        & demo["channel"].isin(channels)
        & demo["merchant_category"].isin(categories)
        & demo["alert_priority_score"].between(score_range[0], score_range[1])
    ].copy()



def kpi_grid(filtered: pd.DataFrame) -> None:
    review_queue = int((filtered["needs_review"] == 1).sum())
    critical = int((filtered["priority_band"] == "critical").sum())
    high = int((filtered["priority_band"] == "high").sum())
    avg_priority = filtered["alert_priority_score"].mean()
    fraud_rate = filtered["Class"].mean() * 100
    avg_amount = filtered["Amount"].mean()

    cards = [
        ("Transactions in scope", f"{len(filtered):,}", "Current filtered monitoring volume", "Live feed"),
        ("Review queue", f"{review_queue:,}", "Transactions above review threshold", "Analyst workload"),
        ("Critical alerts", f"{critical:,}", "Immediate escalation candidates", "Top urgency"),
        ("High alerts", f"{high:,}", "Urgent analyst queue", "Priority tier"),
        ("Avg priority score", f"{avg_priority:,.1f}", "Composite risk + anomaly score", "Risk intensity"),
        ("Avg transaction amount", f"${avg_amount:,.2f}", "Mean monitored amount in current scope", "Exposure signal"),
    ]
    cols = st.columns(6)
    for idx, (label, value, sub, chip) in enumerate(cards):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                    <div class="metric-chip">{chip}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div class='table-note'>Observed fraud reference rate in filtered scope: <strong>{fraud_rate:.2f}%</strong></div>",
        unsafe_allow_html=True,
    )



def format_priority_band(value: str) -> str:
    cls = f"priority-{value}" if value in PRIORITY_ORDER else "priority-low"
    return f"<span class='priority-pill {cls}'>{value.upper()}</span>"



def build_queue_display(queue: pd.DataFrame) -> pd.DataFrame:
    display = queue.copy()
    for col in ["alert_priority_score", "fraud_probability", "anomaly_score", "Amount"]:
        if col in display.columns:
            display[col] = display[col].astype(float)
    display = display.sort_values(["alert_priority_score", "Amount"], ascending=[False, False]).head(200)
    cols = [
        "transaction_id", "event_ts", "customer_id", "Amount", "priority_band", "alert_priority_score",
        "fraud_probability", "anomaly_score", "channel", "merchant_category", "rule_hits", "recommended_action"
    ]
    return display[cols]



def plot_style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.25)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig



def overview_page(filtered: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Executive Overview</div>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["Command Center", "Segment View"])

    with t1:
        c1, c2 = st.columns([1.45, 1])
        with c1:
            hourly = (
                filtered.assign(hour_bucket=filtered["event_ts"].dt.floor("h"))
                .groupby("hour_bucket", as_index=False)
                .agg(alert_volume=("transaction_id", "count"), avg_priority=("alert_priority_score", "mean"))
                .tail(48)
            )
            fig = px.line(
                hourly,
                x="hour_bucket",
                y="alert_volume",
                markers=True,
                title="Alert volume over the last 48 hourly buckets",
            )
            st.plotly_chart(plot_style(fig, 360), use_container_width=True)

        with c2:
            band = filtered["priority_band"].value_counts().reindex(PRIORITY_ORDER).fillna(0).reset_index()
            band.columns = ["priority_band", "count"]
            fig = px.bar(
                band,
                x="priority_band",
                y="count",
                color="priority_band",
                color_discrete_map=PRIORITY_COLORS,
                title="Priority distribution",
            )
            st.plotly_chart(plot_style(fig, 360), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            top_countries = (
                filtered.groupby("country", as_index=False)["alert_priority_score"]
                .mean()
                .sort_values("alert_priority_score", ascending=False)
                .head(10)
            )
            fig = px.bar(
                top_countries,
                x="country",
                y="alert_priority_score",
                title="Highest average priority by country",
            )
            st.plotly_chart(plot_style(fig, 340), use_container_width=True)

        with c4:
            risk_view = (
                filtered.groupby("merchant_category", as_index=False)["Class"]
                .mean()
                .sort_values("Class", ascending=False)
            )
            fig = px.bar(
                risk_view,
                x="merchant_category",
                y="Class",
                title="Observed fraud rate by merchant category",
            )
            st.plotly_chart(plot_style(fig, 340), use_container_width=True)

    with t2:
        s1, s2 = st.columns([1.1, 1])
        with s1:
            channel_summary = (
                filtered.groupby("channel", as_index=False)
                .agg(
                    transactions=("transaction_id", "count"),
                    avg_score=("alert_priority_score", "mean"),
                    fraud_rate=("Class", "mean"),
                )
                .sort_values(["avg_score", "transactions"], ascending=[False, False])
            )
            fig = px.scatter(
                channel_summary,
                x="transactions",
                y="avg_score",
                size="fraud_rate",
                color="channel",
                title="Channel mix: volume vs average priority",
                hover_data={"fraud_rate": ":.2%"},
            )
            st.plotly_chart(plot_style(fig, 360), use_container_width=True)
        with s2:
            score_hist = px.histogram(
                filtered,
                x="alert_priority_score",
                nbins=30,
                title="Alert score distribution",
            )
            st.plotly_chart(plot_style(score_hist, 360), use_container_width=True)

    recent_cols = [
        "transaction_id", "event_ts", "customer_id", "Amount", "channel", "merchant_category",
        "priority_band", "alert_priority_score", "fraud_probability", "rule_hits", "risk_summary"
    ]
    recent = filtered.sort_values(["alert_priority_score", "Amount"], ascending=[False, False])[recent_cols].head(30).copy()
    recent["priority_band"] = recent["priority_band"].str.upper()
    recent["fraud_probability"] = recent["fraud_probability"].map(lambda x: f"{x:.2%}")
    recent["alert_priority_score"] = recent["alert_priority_score"].map(lambda x: f"{x:.1f}")
    recent["Amount"] = recent["Amount"].map(lambda x: f"${x:,.2f}")

    st.markdown("<div class='section-title'>Recent high-signal transactions</div>", unsafe_allow_html=True)
    st.dataframe(recent, use_container_width=True, hide_index=True)



def alert_queue_page(queue: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Analyst Alert Queue</div>", unsafe_allow_html=True)
    st.caption("Sorted by composite alert priority score with triage context for operations review.")

    q1, q2, q3 = st.columns(3)
    q1.metric("Critical in queue", int((queue["priority_band"] == "critical").sum()))
    q2.metric("Avg probability", f"{queue['fraud_probability'].mean():.2%}")
    q3.metric("Avg anomaly score", f"{queue['anomaly_score'].mean():.2f}")

    display = build_queue_display(queue)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "event_ts": st.column_config.DatetimeColumn("event_ts", format="YYYY-MM-DD HH:mm"),
            "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "alert_priority_score": st.column_config.ProgressColumn("alert_priority_score", min_value=0, max_value=100, format="%.1f"),
            "fraud_probability": st.column_config.ProgressColumn("fraud_probability", min_value=0.0, max_value=1.0, format="%.2f"),
            "anomaly_score": st.column_config.NumberColumn("anomaly_score", format="%.2f"),
        },
    )
    st.download_button(
        label="Download alert queue CSV",
        data=queue.to_csv(index=False).encode("utf-8"),
        file_name="alert_queue.csv",
        mime="text/csv",
    )



def investigation_page(filtered: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Investigation Workbench</div>", unsafe_allow_html=True)
    top_ids = filtered.sort_values("alert_priority_score", ascending=False)["transaction_id"].head(100).tolist()
    selected = st.selectbox("Select transaction", options=top_ids)
    row = filtered.loc[filtered["transaction_id"] == selected].iloc[0]

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Alert score", f"{row['alert_priority_score']:.2f}")
    i2.metric("Fraud probability", f"{row['fraud_probability']:.2%}")
    i3.metric("Anomaly score", f"{row['anomaly_score']:.2f}")
    i4.metric("Amount", f"${row['Amount']:,.2f}")

    p1, p2 = st.columns([1.05, 1])
    with p1:
        profile = pd.DataFrame(
            {
                "Field": [
                    "Transaction ID", "Customer ID", "Timestamp", "Amount", "Channel", "Merchant Category",
                    "Country", "Home Country", "Risk Segment", "Priority Band", "Rule Hits", "Recommended Action"
                ],
                "Value": [
                    row["transaction_id"], row["customer_id"], row["event_ts"], f"${row['Amount']:.2f}", row["channel"],
                    row["merchant_category"], row["country"], row["home_country"], row["risk_segment"],
                    row["priority_band"].upper(), int(row["rule_hits"]), row["recommended_action"]
                ],
            }
        )
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("**Transaction profile**")
        st.dataframe(profile, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with p2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("**Why this alert surfaced**")
        st.info(row["risk_summary"])
        st.write("**Triggered rules:**", row["rule_codes"] or "None")
        st.write("**Rule reasons:**", row["rule_reasons"] or "No business rule fired")
        st.markdown("</div>", unsafe_allow_html=True)

    customer_history = filtered[filtered["customer_id"] == row["customer_id"]].sort_values("event_ts").tail(20)
    if not customer_history.empty:
        fig = px.line(
            customer_history,
            x="event_ts",
            y="Amount",
            markers=True,
            title="Recent transaction amounts for this customer",
        )
        st.plotly_chart(plot_style(fig, 340), use_container_width=True)

    if "case_notes" not in st.session_state:
        st.session_state.case_notes = {}
    current_note = st.session_state.case_notes.get(selected, "")

    f1, f2 = st.columns([1.2, 1])
    with f1:
        note = st.text_area("Analyst notes", value=current_note, placeholder="Add a short review note or disposition...")
    with f2:
        status = st.selectbox("Case status", options=["open", "reviewing", "escalated", "closed"])
        st.markdown(
            f"<div class='sidebar-box'><div class='sidebar-title'>Recommended next step</div><div class='sidebar-sub'>{row['recommended_action']}</div></div>",
            unsafe_allow_html=True,
        )
    if st.button("Save case note"):
        st.session_state.case_notes[selected] = f"[{status}] {note}"
        st.success("Case note stored in this session.")



def governance_page(demo: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Model Governance</div>", unsafe_allow_html=True)
    model = cached_model()

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("ROC-AUC", f"{model.metrics.get('roc_auc', 0):.3f}")
    g2.metric("PR-AUC", f"{model.metrics.get('average_precision', 0):.3f}")
    g3.metric("Decision threshold", f"{model.metrics.get('chosen_threshold', 0):.3f}")
    g4.metric("Artifacts loaded", "Yes")

    upper, lower = st.columns([1.15, 1])
    with upper:
        band_perf = (
            demo.groupby("priority_band", as_index=False)
            .agg(
                transaction_count=("transaction_id", "count"),
                observed_fraud_rate=("Class", "mean"),
                avg_probability=("fraud_probability", "mean"),
            )
        )
        fig = px.scatter(
            band_perf,
            x="avg_probability",
            y="observed_fraud_rate",
            size="transaction_count",
            color="priority_band",
            color_discrete_map=PRIORITY_COLORS,
            title="Calibration by priority band",
        )
        st.plotly_chart(plot_style(fig, 360), use_container_width=True)

    with lower:
        feature_table = pd.DataFrame(
            {
                "Feature block": [
                    "Original transaction features",
                    "Operational context",
                    "Behavioral features",
                    "Rules layer",
                    "Priority score",
                ],
                "Description": [
                    "Time, Amount, and anonymized V1-V28 transaction signals.",
                    "Channel, merchant category, device risk, country, and account age context.",
                    "Amount-vs-profile, time since previous transaction, fraud history, and velocity proxies.",
                    "Human-readable business triggers used for analyst trust and prioritization.",
                    "Composite score mixing model probability, anomaly score, amount impact, and rule hits.",
                ],
            }
        )
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Batch Scoring</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an enriched CSV with the expected model schema", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        scored = cached_model().score(frame)
        preview_cols = [
            "Amount", "channel", "merchant_category", "priority_band", "alert_priority_score", "fraud_probability", "risk_summary"
        ]
        st.dataframe(scored[preview_cols].head(50), use_container_width=True, hide_index=True)
        st.download_button(
            label="Download scored results",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="scored_feed.csv",
            mime="text/csv",
        )
    else:
        template_path = DATA_DIR / "demo_input_template.csv"
        if template_path.exists():
            st.download_button(
                label="Download input template",
                data=template_path.read_bytes(),
                file_name="demo_input_template.csv",
                mime="text/csv",
            )



def main() -> None:
    inject_styles()
    demo = load_demo()
    page, bands, channels, categories, score_range = render_sidebar(demo)
    filtered = filter_demo(demo, bands, channels, categories, score_range)
    render_header(filtered)
    kpi_grid(filtered)

    if page == "Executive Overview":
        overview_page(filtered)
    elif page == "Alert Queue":
        alert_queue_page(load_alert_queue())
    elif page == "Investigation Workbench":
        investigation_page(filtered)
    else:
        governance_page(demo)


if __name__ == "__main__":
    main()
