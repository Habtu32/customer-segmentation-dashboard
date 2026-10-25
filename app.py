"""
Customer Segmentation and Personalization Engine - interactive Streamlit dashboard
Team Data Guardians - Florida University Southeast (FUSE) capstone

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engine as E

APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "assets" / "fuse_logo.png"

GREEN, DEEP, GOLD, GOLD_LIGHT = "#1F5D3A", "#14422A", "#C9A227", "#EBCB6B"
INK, MUTED, LINE, CARD, PAGE = "#1F2A24", "#5A675F", "#D9E2D6", "#FFFFFF", "#F4F7F2"
CLUSTER_COLORS = {name: meta["color"] for name, meta in E.CLUSTER_META.items()}
FONT = "Inter, 'Segoe UI', Arial, sans-serif"

st.set_page_config(
    page_title="Data Guardians | Customer Segmentation Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {FONT}; color: {INK}; }}
.stApp {{ background: {PAGE}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #EAF0E7; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {GREEN}; font-family: 'Libre Baskerville', 'Times New Roman', serif; }}

.hero {{ background: linear-gradient(120deg, {DEEP} 0%, {GREEN} 60%, #2E7A4F 100%); border-radius: 22px;
  padding: 30px 38px; display: flex; align-items: center; gap: 30px; position: relative; overflow: hidden;
  box-shadow: 0 14px 34px rgba(20,66,42,.22); margin-bottom: 18px; }}
.hero::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 6px; background: {GOLD}; }}
.hero::before {{ content: ""; position: absolute; right: -80px; top: -120px; width: 380px; height: 380px; border-radius: 50%;
  background: radial-gradient(circle, rgba(235,203,107,.28), rgba(235,203,107,0) 70%); }}
.hero img {{ height: 104px; background: #FDFDFB; border-radius: 16px; padding: 8px 10px; }}
.hero .eyebrow {{ color: {GOLD_LIGHT}; font-size: 12.5px; letter-spacing: 2.2px; font-weight: 600; text-transform: uppercase; }}
.hero h1 {{ color: #FDFDFB; font-family: 'Libre Baskerville', 'Times New Roman', serif; font-size: 34px; line-height: 1.15;
  margin: 6px 0 8px 0; padding: 0; }}
.hero p {{ color: #DCEBE0; margin: 0; font-size: 15px; max-width: 880px; }}
.hero .pill {{ display: inline-block; margin: 12px 8px 0 0; padding: 4px 12px; border-radius: 999px; font-size: 12.5px;
  background: rgba(255,255,255,.12); color: #FDFDFB; border: 1px solid rgba(255,255,255,.22); }}

.demo-banner {{ background: #FFF6D8; border: 1px solid {GOLD}; border-left: 8px solid {GOLD}; color: #5B4A00;
  padding: 12px 18px; border-radius: 12px; margin-bottom: 14px; font-size: 14.5px; }}

.kpi {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 16px 18px 14px 18px; height: 100%;
  box-shadow: 0 2px 10px rgba(31,93,58,.06); transition: transform .15s ease, box-shadow .15s ease; }}
.kpi:hover {{ transform: translateY(-3px); box-shadow: 0 10px 24px rgba(31,93,58,.14); }}
.kpi .l {{ font-size: 12px; letter-spacing: 1.2px; text-transform: uppercase; color: {MUTED}; font-weight: 600; }}
.kpi .v {{ font-family: 'Libre Baskerville', 'Times New Roman', serif; font-size: 30px; font-weight: 700; color: {GREEN};
  line-height: 1.2; margin-top: 4px; }}
.kpi .s {{ font-size: 12.5px; color: {MUTED}; margin-top: 2px; }}

.card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 22px; margin-bottom: 14px;
  box-shadow: 0 2px 10px rgba(31,93,58,.05); }}
.card h4 {{ margin: 0 0 6px 0; color: {GREEN}; font-family: 'Libre Baskerville', 'Times New Roman', serif; font-size: 18px; }}
.card p {{ margin: 0; color: #2F3B34; font-size: 14.5px; line-height: 1.5; }}
.insight {{ background: {CARD}; border: 1px solid {LINE}; border-left: 7px solid {GOLD}; border-radius: 14px;
  padding: 14px 18px; margin-bottom: 12px; }}
.insight b {{ color: {GREEN}; }}
.insight .t {{ font-size: 12px; letter-spacing: 1.3px; text-transform: uppercase; color: {MUTED}; font-weight: 600; }}
.insight .b {{ font-size: 15.5px; color: {INK}; margin-top: 3px; line-height: 1.45; }}

.section-title {{ font-family: 'Libre Baskerville', 'Times New Roman', serif; color: {GREEN}; font-size: 22px; margin: 8px 0 2px 0; }}
.section-sub {{ color: {MUTED}; font-size: 14px; margin-bottom: 10px; }}

.strategy {{ background: linear-gradient(135deg, {DEEP}, {GREEN}); color: #FDFDFB; border-radius: 18px; padding: 22px 26px;
  border-bottom: 6px solid {GOLD}; box-shadow: 0 10px 26px rgba(20,66,42,.2); }}
.strategy .t {{ color: {GOLD_LIGHT}; font-size: 12.5px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; }}
.strategy .a {{ font-family: 'Libre Baskerville', 'Times New Roman', serif; font-size: 25px; line-height: 1.3; margin: 8px 0 12px 0; }}
.strategy .m {{ color: #DCEBE0; font-size: 14px; }}
.chip {{ display: inline-block; background: {GOLD}; color: {INK}; padding: 3px 12px; border-radius: 999px; font-size: 13px;
  font-weight: 600; margin: 4px 6px 0 0; }}
.chip.soft {{ background: #E8F0EA; color: {GREEN}; border: 1px solid {LINE}; }}

table.matrix {{ width: 100%; border-collapse: separate; border-spacing: 8px; table-layout: fixed; }}
table.matrix th {{ color: {GREEN}; font-size: 14px; text-align: left; padding: 4px 6px; }}
table.matrix td {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 12px 14px; vertical-align: top;
  font-size: 13.5px; line-height: 1.4; color: #2F3B34; }}
table.matrix td.sel {{ border: 2px solid {GOLD}; background: #FFFBEA; box-shadow: 0 6px 18px rgba(201,162,39,.25); }}
table.matrix th.row span {{ display:block; color:{MUTED}; font-weight:500; font-size:12px; }}

button[data-baseweb="tab"] {{ font-weight: 600; font-size: 15px; padding: 10px 16px; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {GREEN}; }}
div[data-baseweb="tab-highlight"] {{ background-color: {GOLD} !important; height: 4px !important; }}
.stDownloadButton button {{ background: {GREEN}; color: #fff; border: 0; border-radius: 10px; padding: 8px 18px; }}
.stDownloadButton button:hover {{ background: {DEEP}; color: #fff; }}
.footer {{ text-align: center; color: {MUTED}; font-size: 13px; margin-top: 30px; padding-top: 14px; border-top: 1px solid {LINE}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def esc(text: object) -> str:
    return html.escape(str(text))


def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.1f}K"
    return f"${x:,.0f}"


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>'


def show(fig: go.Figure, key: str | None = None) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def style(fig: go.Figure, height: int = 400, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=56, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK),
        title=dict(font=dict(size=16, color=GREEN, family="Libre Baskerville, Times New Roman, serif"), x=0.0),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=13, font_family=FONT, bordercolor=LINE),
    )
    fig.update_xaxes(gridcolor="#E5ECE2", zerolinecolor="#E5ECE2", linecolor=LINE)
    fig.update_yaxes(gridcolor="#E5ECE2", zerolinecolor="#E5ECE2", linecolor=LINE)
    return fig


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="section-title">{esc(title)}</div><div class="section-sub">{esc(sub)}</div>', unsafe_allow_html=True)


def profile_of(view: pd.DataFrame) -> pd.DataFrame:
    prof = (
        view.groupby("Cluster", observed=True)
        .agg(
            Customers=("Customer ID", "count"),
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            R=("R", "mean"),
            F=("F", "mean"),
            M=("M", "mean"),
            RFM=("RFM", "mean"),
            Revenue=("Monetary", "sum"),
        )
        .reindex(E.CLUSTER_ORDER)
        .dropna(how="all")
    )
    prof["Pct customers"] = prof["Customers"] / prof["Customers"].sum() * 100
    prof["Pct revenue"] = prof["Revenue"] / prof["Revenue"].sum() * 100
    return prof


def logo_b64() -> str | None:
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return None


# --------------------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading the file...")
def load_uploaded(raw: bytes, name: str) -> pd.DataFrame:
    return E.standardize_columns(E.read_table(raw, name))


@st.cache_data(show_spinner="Generating demo data...")
def load_demo() -> pd.DataFrame:
    return E.standardize_columns(E.make_demo_data())


@st.cache_data(show_spinner="Running RFM scoring and K-Means...")
def analyze(df: pd.DataFrame, date_source: str) -> E.Result:
    return E.run_pipeline(df, date_source)


@st.cache_data(show_spinner="Checking model quality...")
def diagnostics(scaled: np.ndarray, labels: np.ndarray) -> dict:
    return E.model_diagnostics(scaled, labels)


def find_local_file() -> Path | None:
    for folder in (APP_DIR, APP_DIR / "data"):
        if folder.exists():
            for path in sorted(folder.iterdir()):
                if "superstore" in path.name.lower() and path.suffix.lower() in (".csv", ".xlsx", ".xls"):
                    return path
    return None


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=110)
    st.markdown("### Data source")
    uploaded = st.file_uploader("Global Superstore file (CSV or Excel)", type=["csv", "xlsx", "xls"])
    local_file = find_local_file()

    is_demo = False
    source_label = ""
    try:
        if uploaded is not None:
            raw_df = load_uploaded(uploaded.getvalue(), uploaded.name)
            source_label = uploaded.name
        elif local_file is not None:
            raw_df = load_uploaded(local_file.read_bytes(), local_file.name)
            source_label = local_file.name
        else:
            raw_df = load_demo()
            is_demo = True
            source_label = "Synthetic demo data"
    except E.DataError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the file: {exc}")
        st.stop()

    has_week = {"Year", "weeknum"} <= set(raw_df.columns)
    date_choice = "Rebuild from Year + weeknum (report method)"
    if has_week and "Order Date" in raw_df.columns:
        date_choice = st.radio(
            "Order date source",
            ["Rebuild from Year + weeknum (report method)", "Use the Order Date column as-is"],
            help="The Global Superstore Order Date field is corrupted (every record has the same date), "
            "so the report rebuilds dates as the Monday of each recorded week.",
        )
    date_source = "rebuilt" if (has_week and date_choice.startswith("Rebuild")) else "original"
    st.caption(f"Source: {source_label}")

try:
    result = analyze(raw_df, date_source)
except E.DataError as exc:
    st.error(str(exc))
    st.stop()

customers = result.customers
with st.sidebar:
    st.markdown("### Filters")
    seg_options = [s for s in E.SEGMENT_ORDER if s in set(customers["Segment"])] or sorted(customers["Segment"].unique())
    sel_segments = st.multiselect("Account type", seg_options, default=seg_options)
    sel_clusters = st.multiselect("Customer cluster", E.CLUSTER_ORDER, default=E.CLUSTER_ORDER)
    sel_rfm = st.slider("RFM total score", 3, 15, (3, 15), help="Sum of the three 1-5 scores.")
    max_rec = int(customers["Recency"].max())
    sel_rec = st.slider("Days since last order", 0, max_rec, (0, max_rec))
    st.markdown("---")
    st.caption(
        "Reference date: **{}**  \nCustomers: **{:,}**  \nOrder records: **{:,}**".format(
            result.reference_date.strftime("%d %b %Y"), len(customers), len(result.transactions)
        )
    )

mask = (
    customers["Segment"].isin(sel_segments)
    & customers["Cluster"].isin(sel_clusters)
    & customers["RFM"].between(*sel_rfm)
    & customers["Recency"].between(*sel_rec)
)
view = customers[mask].copy()
tx_view = result.transactions[result.transactions["Customer ID"].isin(view["Customer ID"])]

# --------------------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------------------
logo = logo_b64()
logo_html = f'<img src="data:image/png;base64,{logo}" alt="FUSE logo">' if logo else ""
st.markdown(
    f"""
<div class="hero">
  {logo_html}
  <div>
    <div class="eyebrow">Florida University Southeast · Capstone · BI &amp; Data Analytics</div>
    <h1>Customer Segmentation &amp; Personalization Engine</h1>
    <p>RFM analysis and K-Means clustering turn order records into four customer segments and twelve
       segment-specific actions for an online retail store.</p>
    <span class="pill">Team Data Guardians</span><span class="pill">Global Superstore data</span><span class="pill">K = 4 · RFM 3-15</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
if is_demo:
    st.markdown(
        '<div class="demo-banner"><b>Demo data.</b> These customers are synthetic and only imitate the report. '
        "Upload the real Global Superstore file in the sidebar (or put it next to <code>app.py</code>) to see real results.</div>",
        unsafe_allow_html=True,
    )

if view.empty:
    st.warning("No customers match the current filters. Widen the filters in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------------------
prof = profile_of(view)
top2_share = prof.loc[prof.index.isin(["Champions", "At-Risk"]), "Pct revenue"].sum()
top2_cust = prof.loc[prof.index.isin(["Champions", "At-Risk"]), "Pct customers"].sum()
cols = st.columns(6)
cards = [
    ("Customers", f"{len(view):,}", f"{len(view) / len(customers) * 100:.0f}% of all accounts"),
    ("Order records", f"{len(tx_view):,}", f"{len(tx_view) / max(len(view), 1):.1f} per customer"),
    ("Revenue", money(view["Monetary"].sum()), f"{money(view['Monetary'].mean())} per customer"),
    ("Average RFM score", f"{view['RFM'].mean():.1f}", "scale 3 to 15"),
    ("Champions + At-Risk", f"{top2_share:.1f}%", f"of revenue from {top2_cust:.0f}% of customers"),
    ("Silhouette", f"{result.silhouette:.2f}", "cluster separation (moderate)"),
]
for col, (label, value, sub) in zip(cols, cards):
    col.markdown(kpi(label, value, sub), unsafe_allow_html=True)
st.write("")

tab_over, tab_seg, tab_acc, tab_pers, tab_cust, tab_model = st.tabs(
    ["Overview", "Segments", "Account types", "Personalization", "Customer explorer", "Model & data quality"]
)

# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------
with tab_over:
    section("Who are our customers, and where does the money come from?",
            "Share of customers versus share of revenue for each behavioral cluster.")
    c1, c2 = st.columns([1, 1.35])
    with c1:
        fig = go.Figure(
            go.Pie(
                labels=prof.index.tolist(),
                values=prof["Customers"],
                hole=0.62,
                sort=False,
                marker=dict(colors=[CLUSTER_COLORS[c] for c in prof.index], line=dict(color="#FFFFFF", width=3)),
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>",
            )
        )
        fig.add_annotation(text=f"<b>{len(view):,}</b><br>customers", showarrow=False, font=dict(size=20, color=GREEN))
        style(fig, 400).update_layout(title_text="Customers by cluster")
        show(fig, "donut")
    with c2:
        order = prof.index.tolist()[::-1]
        fig = go.Figure()
        fig.add_bar(
            y=order, x=prof.loc[order, "Pct customers"], name="% of customers", orientation="h",
            marker_color=GREEN, texttemplate="%{x:.1f}%", textposition="outside",
        )
        fig.add_bar(
            y=order, x=prof.loc[order, "Pct revenue"], name="% of revenue", orientation="h",
            marker_color=GOLD, texttemplate="%{x:.1f}%", textposition="outside",
        )
        style(fig, 400).update_layout(
            barmode="group", title_text="Share of customers versus share of revenue", bargap=0.28,
            xaxis=dict(title="", range=[0, max(prof["Pct customers"].max(), prof["Pct revenue"].max()) * 1.2]),
        )
        show(fig, "bars")

    c3, c4 = st.columns([1.35, 1])
    with c3:
        sun = view.groupby(["Cluster", "Segment"], observed=True)["Monetary"].sum().reset_index()
        sun["Cluster"] = sun["Cluster"].astype(str)
        fig = px.sunburst(
            sun, path=["Cluster", "Segment"], values="Monetary", color="Cluster",
            color_discrete_map=CLUSTER_COLORS,
        )
        fig.update_traces(hovertemplate="<b>%{label}</b><br>$%{value:,.0f} revenue<extra></extra>", marker_line_color="#FFFFFF")
        style(fig, 420, legend=False).update_layout(title_text="Revenue by cluster and account type")
        show(fig, "sunburst")
    with c4:
        st.markdown('<div class="section-title" style="font-size:18px">What the data says</div>', unsafe_allow_html=True)
        biggest = prof["Customers"].idxmax()
        st.markdown(
            f'<div class="insight"><div class="t">Largest group</div><div class="b"><b>{esc(biggest)}</b> is the largest cluster: '
            f'{prof.loc[biggest, "Pct customers"]:.1f}% of customers but {prof.loc[biggest, "Pct revenue"]:.1f}% of revenue.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="insight"><div class="t">Revenue concentration</div><div class="b"><b>Champions + At-Risk</b> are '
            f'{top2_cust:.1f}% of customers and <b>{top2_share:.1f}%</b> of revenue.</div></div>',
            unsafe_allow_html=True,
        )
        if {"Champions", "At-Risk"} <= set(prof.index):
            ratio = prof.loc["At-Risk", "Recency"] / prof.loc["Champions", "Recency"]
            st.markdown(
                f'<div class="insight"><div class="t">Retention signal</div><div class="b"><b>At-Risk</b> customers have been silent '
                f'<b>{ratio:.1f}×</b> longer than Champions ({prof.loc["At-Risk", "Recency"]:.0f} vs '
                f'{prof.loc["Champions", "Recency"]:.0f} days). Re-engage them first.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Select both Champions and At-Risk to compare their silence.")

# --------------------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------------------
with tab_seg:
    section("Cluster profiles", "Mean behavior of each cluster. Revenue share = cluster revenue divided by total revenue.")
    table = prof.reset_index()[["Cluster", "Customers", "Pct customers", "Recency", "Frequency", "Monetary", "RFM", "Pct revenue"]]
    table["Cluster"] = table["Cluster"].astype(str)
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Customers": st.column_config.NumberColumn("Customers", format="%d"),
            "Pct customers": st.column_config.ProgressColumn("% of customers", min_value=0, max_value=100, format="%.1f%%"),
            "Recency": st.column_config.NumberColumn("Recency (days)", format="%.1f"),
            "Frequency": st.column_config.NumberColumn("Frequency (records)", format="%.1f"),
            "Monetary": st.column_config.NumberColumn("Monetary ($)", format="$%.0f"),
            "RFM": st.column_config.NumberColumn("Mean RFM total", format="%.1f"),
            "Pct revenue": st.column_config.ProgressColumn("% of revenue", min_value=0, max_value=100, format="%.1f%%"),
        },
    )
    c1, c2 = st.columns([1, 1.25])
    with c1:
        theta = ["Recency score", "Frequency score", "Monetary score", "Recency score"]
        fig = go.Figure()
        for name in prof.index:
            r = [prof.loc[name, "R"], prof.loc[name, "F"], prof.loc[name, "M"], prof.loc[name, "R"]]
            fig.add_trace(
                go.Scatterpolar(r=r, theta=theta, name=name, fill="toself", opacity=0.55,
                                line=dict(color=CLUSTER_COLORS[name], width=2.5),
                                hovertemplate="%{theta}: %{r:.2f}<extra>" + name + "</extra>")
            )
        style(fig, 470).update_layout(
            title_text="Average RFM scores (1 to 5)",
            polar=dict(radialaxis=dict(range=[0, 5], gridcolor="#E5ECE2"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=70, r=70, t=70, b=30),
        )
        show(fig, "radar")
    with c2:
        axes = {"Recency (days)": "Recency", "Frequency (records)": "Frequency", "Monetary ($)": "Monetary"}
        ca, cb, cc = st.columns(3)
        xlab = ca.selectbox("X axis", list(axes), index=0)
        ylab = cb.selectbox("Y axis", list(axes), index=2)
        logy = cc.checkbox("Log scale on Y", value=True)
        sample = view.sample(min(len(view), 3000), random_state=1)
        fig = px.scatter(
            sample, x=axes[xlab], y=axes[ylab], color=sample["Cluster"].astype(str), size="Frequency",
            size_max=16, color_discrete_map=CLUSTER_COLORS, opacity=0.72,
            hover_name="Customer Name", hover_data={"Customer ID": True, "Segment": True, "RFM": True},
            category_orders={"color": E.CLUSTER_ORDER}, log_y=logy,
        )
        style(fig, 430).update_layout(title_text=f"{ylab} against {xlab} (up to 3,000 customers)", legend_title_text="")
        fig.update_xaxes(title=xlab)
        fig.update_yaxes(title=ylab)
        show(fig, "scatter")
    metric = st.radio("Distribution of", list(axes), horizontal=True)
    fig = px.box(
        view.assign(ClusterName=view["Cluster"].astype(str)), x="ClusterName", y=axes[metric], color="ClusterName",
        color_discrete_map=CLUSTER_COLORS, points=False,
        category_orders={"ClusterName": E.CLUSTER_ORDER}, log_y=(axes[metric] != "Recency"),
    )
    style(fig, 380, legend=False).update_layout(title_text=f"{metric} by cluster", xaxis_title="", yaxis_title=metric)
    show(fig, "box")

# --------------------------------------------------------------------------------------
# Account types
# --------------------------------------------------------------------------------------
with tab_acc:
    section("Does behavior depend on account type?",
            "Cross-tabulation of behavioral cluster against Consumer, Corporate and Home Office accounts.")
    cross = pd.crosstab(view["Cluster"], view["Segment"]).reindex(E.CLUSTER_ORDER).dropna(how="all")
    cross = cross.reindex(columns=[c for c in E.SEGMENT_ORDER if c in cross.columns])
    pct_row = cross.div(cross.sum(axis=1), axis=0) * 100
    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(
            pct_row.values, x=pct_row.columns, y=[str(i) for i in pct_row.index], text_auto=".1f",
            color_continuous_scale=[[0, "#EEF4EC"], [1, GREEN]], aspect="auto",
        )
        style(fig, 380, legend=False).update_layout(title_text="Account-type mix inside each cluster (%)", coloraxis_showscale=False)
        fig.update_traces(hovertemplate="%{y} · %{x}<br>%{z:.1f}% of the cluster<extra></extra>")
        show(fig, "heat")
    with c2:
        long = cross.reset_index().melt(id_vars="Cluster", var_name="Account type", value_name="Customers")
        long["Cluster"] = long["Cluster"].astype(str)
        fig = px.bar(
            long, x="Customers", y="Cluster", color="Account type", orientation="h", barmode="stack",
            color_discrete_map={"Consumer": GREEN, "Corporate": GOLD, "Home Office": "#7A8795"},
            category_orders={"Cluster": E.CLUSTER_ORDER[::-1], "Account type": E.SEGMENT_ORDER},
        )
        style(fig, 380).update_layout(title_text="Customers by cluster and account type", yaxis_title="", legend_title_text="")
        show(fig, "stack")
    chi = E.chi_square(customers)
    verdict = (
        "No evidence that cluster membership depends on account type. Each cluster has roughly the same Consumer, "
        "Corporate and Home Office mix, and the effect size is negligible."
        if chi["p"] > 0.05
        else "Cluster membership and account type are statistically related in this data; check the mix above."
    )
    st.markdown(
        f"""<div class="card"><h4>Chi-square test of independence (all customers)</h4>
        <p><b>χ²({chi['dof']}, N = {chi['n']:,}) = {chi['chi2']:.2f}, p = {chi['p']:.3f}, Cramér's V = {chi['v']:.2f}.</b><br>{esc(verdict)}</p>
        <p style="margin-top:8px"><b>Use both dimensions:</b> behavior (the cluster) decides the goal and urgency; account type decides the channel and offer.</p></div>""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# Personalization
# --------------------------------------------------------------------------------------
with tab_pers:
    section("The personalization matrix", "4 clusters × 3 account types = 12 strategies. Pick a cell to see its action and leading product sub-categories.")
    pc1, pc2 = st.columns(2)
    sel_c = pc1.radio("Cluster", E.CLUSTER_ORDER, horizontal=True, key="pc")
    sel_s = pc2.radio("Account type", E.SEGMENT_ORDER, horizontal=True, key="ps")
    focus = result.focus
    top2: dict[tuple[str, str], list[str]] = {}
    if focus is not None:
        for (cl, ac), grp in focus[focus["Rank"] <= 2].groupby(["Cluster", "Account"], observed=True):
            top2[(str(cl), str(ac))] = grp.sort_values("Rank")["Sub-Category"].tolist()

    head = "".join(f"<th>{esc(s)}</th>" for s in E.SEGMENT_ORDER)
    body = ""
    for cl in E.CLUSTER_ORDER:
        body += f'<tr><th class="row">{esc(cl)}<span>{esc(E.CLUSTER_META[cl]["goal"])}</span></th>'
        for ac in E.SEGMENT_ORDER:
            chips = "".join(f'<span class="chip soft">{esc(s)}</span>' for s in top2.get((cl, ac), []))
            cls = ' class="sel"' if (cl, ac) == (sel_c, sel_s) else ""
            body += f"<td{cls}>{esc(E.STRATEGIES[(cl, ac)])}<div>{chips}</div></td>"
        body += "</tr>"
    st.markdown(f'<table class="matrix"><tr><th></th>{head}</tr>{body}</table>', unsafe_allow_html=True)

    st.write("")
    cell_cust = view[(view["Cluster"] == sel_c) & (view["Segment"] == sel_s)]
    left, right = st.columns([1.15, 1])
    with left:
        chips = "".join(f'<span class="chip">{esc(s)}</span>' for s in top2.get((sel_c, sel_s), []))
        meta = E.CLUSTER_META[sel_c]
        st.markdown(
            f"""<div class="strategy"><div class="t">{esc(sel_c)} · {esc(sel_s)} · {esc(meta['goal'])}</div>
            <div class="a">{esc(E.STRATEGIES[(sel_c, sel_s)])}</div>
            <div class="m"><b>Lead with:</b> {chips if chips else 'Sub-category data not in this file'}</div>
            <div class="m" style="margin-top:12px"><b>Measure success by:</b> {esc(meta['measures'])}</div></div>""",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.markdown(kpi("Customers in cell", f"{len(cell_cust):,}", "after filters"), unsafe_allow_html=True)
        m2.markdown(kpi("Revenue", money(cell_cust["Monetary"].sum()) if len(cell_cust) else "$0", "total sales"), unsafe_allow_html=True)
        m3.markdown(kpi("Avg recency", f"{cell_cust['Recency'].mean():.0f} d" if len(cell_cust) else "-", "days since last order"), unsafe_allow_html=True)
    with right:
        if focus is not None:
            cell_focus = focus[(focus["Cluster"] == sel_c) & (focus["Account"] == sel_s)].sort_values("Rank").head(8)
            cell_focus = cell_focus.assign(Top=np.where(cell_focus["Rank"] <= 2, "Top two", "Other"))
            fig = px.bar(
                cell_focus.iloc[::-1], x="Share", y="Sub-Category", color="Top", orientation="h",
                color_discrete_map={"Top two": GOLD, "Other": "#B8C9BC"},
            )
            style(fig, 360).update_layout(title_text="Sub-categories by share of the cell's sales", xaxis_title="% of cell sales", yaxis_title="", legend_title_text="")
            fig.update_traces(hovertemplate="%{y}: %{x:.1f}% of sales<extra></extra>")
            show(fig, "focus")
        else:
            st.info("This file has no Sub-Category column, so product focus is unavailable.")

    st.markdown("#### Recommended priorities")
    prio = []
    for cl in sorted(E.CLUSTER_ORDER, key=lambda c: E.CLUSTER_META[c]["priority"]):
        row = result.profile.loc[cl]
        prio.append(
            {
                "Priority": E.CLUSTER_META[cl]["priority"],
                "Cluster": cl,
                "Customers": int(row["Customers"]),
                "% revenue": float(row["Pct revenue"]),
                "Recommended focus": E.CLUSTER_META[cl]["focus"],
                "Suggested measures": E.CLUSTER_META[cl]["measures"],
            }
        )
    st.dataframe(
        pd.DataFrame(prio), hide_index=True,
        column_config={
            "% revenue": st.column_config.ProgressColumn("% of revenue", min_value=0, max_value=100, format="%.1f%%"),
            "Recommended focus": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption("These are proposed actions. They have not been tested on customers; run a controlled pilot with a comparison group first.")
    matrix_rows = []
    for cl in E.CLUSTER_ORDER:
        for ac in E.SEGMENT_ORDER:
            tops = top2.get((cl, ac), ["", ""]) + ["", ""]
            matrix_rows.append(
                {"Cluster": cl, "Account type": ac, "Strategy": E.STRATEGIES[(cl, ac)],
                 "Focus 1": tops[0], "Focus 2": tops[1],
                 "Customers": int(((customers["Cluster"] == cl) & (customers["Segment"] == ac)).sum())}
            )
    st.download_button("Download the 12-strategy matrix (CSV)", pd.DataFrame(matrix_rows).to_csv(index=False).encode(),
                       file_name="personalization_matrix.csv", mime="text/csv")

# --------------------------------------------------------------------------------------
# Customer explorer
# --------------------------------------------------------------------------------------
with tab_cust:
    section("Find a customer and see the recommended action", "Search by Customer ID or name. The table follows the sidebar filters.")
    q = st.text_input("Search", placeholder="Type part of a Customer ID or name")
    found = view
    if q.strip():
        needle = q.strip().lower()
        found = view[view["Customer ID"].str.lower().str.contains(needle, regex=False)
                     | view["Customer Name"].astype(str).str.lower().str.contains(needle, regex=False)]
    st.caption(f"{len(found):,} customers")
    cols_show = ["Customer ID", "Customer Name", "Segment", "Cluster", "Recency", "Frequency", "Monetary", "R", "F", "M", "RFM", "Action"]
    shown = found[cols_show].sort_values("Monetary", ascending=False).copy()
    shown["Cluster"] = shown["Cluster"].astype(str)
    st.dataframe(
        shown, hide_index=True, height=360,
        column_config={
            "Segment": "Account type",
            "Recency": st.column_config.NumberColumn("Recency (days)", format="%d"),
            "Frequency": st.column_config.NumberColumn("Frequency", format="%d"),
            "Monetary": st.column_config.NumberColumn("Monetary ($)", format="$%.0f"),
            "RFM": st.column_config.ProgressColumn("RFM total", min_value=3, max_value=15, format="%d"),
            "Action": st.column_config.TextColumn("Recommended action", width="large"),
        },
    )
    st.download_button("Download these customers (CSV)", shown.to_csv(index=False).encode(),
                       file_name="customer_segments.csv", mime="text/csv")

    if len(found):
        st.markdown('<div class="section-title" style="font-size:20px;margin-top:16px">Customer 360</div>', unsafe_allow_html=True)
        options = found.sort_values("Monetary", ascending=False).head(500)
        labels = {r["Customer ID"]: f'{r["Customer Name"]}  ({r["Customer ID"]})' for _, r in options.iterrows()}
        chosen = st.selectbox("Choose a customer", list(labels), format_func=lambda k: labels[k])
        row = customers[customers["Customer ID"] == chosen].iloc[0]
        cluster = str(row["Cluster"])
        g1, g2, g3 = st.columns([1, 1.2, 1.4])
        with g1:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number", value=float(row["RFM"]),
                    number=dict(font=dict(color=GREEN, size=44)),
                    gauge=dict(
                        axis=dict(range=[3, 15], tickcolor=MUTED), bar=dict(color=CLUSTER_COLORS[cluster], thickness=0.34),
                        steps=[dict(range=[3, 7], color="#EEF1F0"), dict(range=[7, 11], color="#E3EBE0"), dict(range=[11, 15], color="#D2E3D5")],
                        borderwidth=0,
                    ),
                    title=dict(text="RFM total (3 to 15)", font=dict(size=14, color=MUTED)),
                )
            )
            style(fig, 280, legend=False)
            show(fig, "gauge")
        with g2:
            st.markdown(
                f"""<div class="card"><h4>{esc(row['Customer Name'])}</h4>
                <p>{esc(row['Customer ID'])} · {esc(row['Segment'])}<br>
                <span class="chip" style="background:{CLUSTER_COLORS[cluster]};color:#fff">{esc(cluster)}</span></p>
                <p style="margin-top:10px">Last order: <b>{row['Recency']:,} days</b> before the reference date<br>
                Order records: <b>{int(row['Frequency']):,}</b><br>Total sales: <b>${row['Monetary']:,.0f}</b><br>
                Scores R / F / M: <b>{int(row['R'])} / {int(row['F'])} / {int(row['M'])}</b></p></div>""",
                unsafe_allow_html=True,
            )
        with g3:
            tops = top2.get((cluster, str(row["Segment"])), [])
            chips = "".join(f'<span class="chip">{esc(s)}</span>' for s in tops)
            st.markdown(
                f"""<div class="strategy"><div class="t">Recommended action</div><div class="a" style="font-size:21px">{esc(row['Action'])}</div>
                <div class="m"><b>Lead with:</b> {chips if chips else 'n/a'}</div></div>""",
                unsafe_allow_html=True,
            )
        history = result.transactions[result.transactions["Customer ID"] == chosen]
        hist = history.groupby("OrderDate")["Sales"].sum().reset_index()
        fig = px.bar(hist, x="OrderDate", y="Sales", color_discrete_sequence=[CLUSTER_COLORS[cluster]])
        style(fig, 260, legend=False).update_layout(title_text="Sales by recorded order week (dates are the Monday of each week)", xaxis_title="", yaxis_title="Sales ($)")
        show(fig, "history")

# --------------------------------------------------------------------------------------
# Model and data quality
# --------------------------------------------------------------------------------------
with tab_model:
    section("How good is the model?", "Elbow curve, silhouette by K, stability across random seeds, and a check against the report.")
    diag = diagnostics(result.scaled, customers["Label"].to_numpy())
    c1, c2 = st.columns(2)
    with c1:
        elbow = diag["elbow"]
        fig = go.Figure()
        fig.add_scatter(x=elbow["K"], y=elbow["Inertia"], mode="lines+markers", line=dict(color=GREEN, width=3),
                        marker=dict(size=9, color="#FFFFFF", line=dict(color=GREEN, width=3)), name="Inertia")
        sel = elbow[elbow["K"] == E.K_CLUSTERS]
        fig.add_scatter(x=sel["K"], y=sel["Inertia"], mode="markers", marker=dict(size=17, color=GOLD, line=dict(color="#FFFFFF", width=2)), name="Selected K = 4")
        style(fig, 360).update_layout(title_text="Elbow curve (inertia)", xaxis=dict(title="Number of clusters K", dtick=1), yaxis_title="Inertia")
        show(fig, "elbow")
    with c2:
        sil = diag["silhouette"]
        fig = px.bar(sil, x="K", y="Silhouette", color_discrete_sequence=["#B8C9BC"], text_auto=".2f")
        fig.update_traces(marker_color=[GOLD if k == E.K_CLUSTERS else "#B8C9BC" for k in sil["K"]])
        style(fig, 360, legend=False).update_layout(title_text="Silhouette score by K", xaxis=dict(dtick=1))
        show(fig, "sil")
    st.caption("The elbow falls steeply up to K = 4 and then declines gradually with no sharp plateau, so K = 4 rests on the curve, "
               "the four marketing goals it supports, and stability. Choosing K is a judgment.")

    s1, s2 = st.columns([1, 1.4])
    with s1:
        stab = diag["stability"].copy()
        st.markdown("##### Stability across random seeds")
        st.dataframe(stab, hide_index=True, column_config={"Adjusted Rand index": st.column_config.ProgressColumn("Adjusted Rand index", min_value=0, max_value=1, format="%.3f")})
        st.caption("1.0 means the same groups as the main solution (seed 42).")
    with s2:
        st.markdown("##### Check against the final report")
        ref = E.REPORT_REFERENCE
        chi_all = E.chi_square(customers)
        rows = [
            ("Order records", f"{len(result.transactions):,}", f"{ref['records']:,}", len(result.transactions) == ref["records"]),
            ("Customer accounts", f"{len(customers):,}", f"{ref['accounts']:,}", len(customers) == ref["accounts"]),
        ]
        for name in E.CLUSTER_ORDER:
            mine = int(result.profile.loc[name, "Customers"])
            rows.append((f"{name} customers", f"{mine:,}", f"{ref['cluster_sizes'][name]:,}", mine == ref["cluster_sizes"][name]))
        rows.append(("Silhouette", f"{result.silhouette:.2f}", f"{ref['silhouette']:.2f}", abs(round(result.silhouette, 2) - ref["silhouette"]) < 0.005))
        rows.append(("Chi-square p-value", f"{chi_all['p']:.3f}", f"{ref['p_value']:.3f}", abs(chi_all["p"] - ref["p_value"]) < 0.005))
        check = pd.DataFrame(rows, columns=["Item", "This app", "Report", "Match"])
        check["Match"] = np.where(check["Match"], "✅ match", "⚠️ differs")
        st.dataframe(check, hide_index=True)
        if is_demo:
            st.caption("Demo data is synthetic, so differences from the report are expected.")
        else:
            st.caption("Small differences can come from the quintile tie-breaking rule or the date source; see the README.")

    section("Data quality", "Why the report rebuilds the Order Date.")
    info = result.date_info
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(kpi("Order records", f"{len(result.transactions):,}", f"{raw_df.shape[1]} columns"), unsafe_allow_html=True)
    if info.get("has_order_date"):
        d2.markdown(kpi("Distinct Order Dates (as-is)", f"{info.get('distinct_original', 0):,}", "1 means the field is corrupted"), unsafe_allow_html=True)
        d3.markdown(kpi("Recency SD (as-is)", f"{info.get('recency_sd_original', float('nan')):.1f}", "0.0 means no customer can be told apart"), unsafe_allow_html=True)
    rec = customers["Recency"]
    d4.markdown(kpi("Recency SD (used)", f"{rec.std(ddof=1):.1f}", f"mean {rec.mean():.1f}, range {rec.min()} to {rec.max()} days"), unsafe_allow_html=True)
    st.markdown(
        f"""<div class="card" style="margin-top:12px"><h4>Method in one paragraph</h4><p>
        Dates are rebuilt as the Monday of each recorded week from Year and week number, so a date can be off by up to 6 days.
        The reference date is {result.reference_date.strftime('%d %B %Y')} (one day after the latest order week). Recency is days since the last order
        (scored in reverse), Frequency counts order records (product lines), and Monetary is total Sales; each is scored 1 to 5 by quintile.
        The three scores are standardized and clustered with K-Means (K = 4, random_state = 42, n_init = 10).</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("##### Limitations to state openly")
    lim = st.columns(3)
    limits = [
        ("Weekly dates", "Rebuilt dates can be off by up to 6 days."),
        ("Moderate separation", "Silhouette near 0.38: the clusters overlap at their edges."),
        ("Quintile scores", "Scores keep rank order but not the size of differences."),
        ("Customer ID", "One ID is treated as one account; some names appear under several IDs."),
        ("Historical data", "The data cover 2011 to 2014: the method transfers, the numbers do not."),
        ("No campaign test", "The strategies are proposed rules; their effect has not been measured."),
    ]
    for i, (title, text) in enumerate(limits):
        lim[i % 3].markdown(f'<div class="card"><h4 style="font-size:16px">{esc(title)}</h4><p>{esc(text)}</p></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Team Data Guardians · Florida University Southeast capstone · Customer Segmentation and Personalization Engine</div>', unsafe_allow_html=True)
