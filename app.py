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
import plotly.io as pio
import streamlit as st

import engine as E

APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "assets" / "fuse_logo.png"

# --------------------------------------------------------------------------------------
# Brand palette (base colors kept from the original theme)
# --------------------------------------------------------------------------------------
GREEN, DEEP, FOREST, GOLD, GOLD_LIGHT = "#1F5D3A", "#14422A", "#0E2F1E", "#C9A227", "#EBCB6B"
SAGE, MINT, SLATE = "#6FB38B", "#E8F1EA", "#7A8795"
INK, MUTED, LINE, CARD, PAGE = "#1F2A24", "#5A675F", "#DDE6DA", "#FFFFFF", "#F4F7F2"
GRID = "#E9EFE6"
CLUSTER_COLORS = {name: meta["color"] for name, meta in E.CLUSTER_META.items()}
ACCOUNT_COLORS = {"Consumer": GREEN, "Corporate": GOLD, "Home Office": SLATE}
FONT = "Inter, 'Segoe UI', Arial, sans-serif"
SERIF = "'Libre Baskerville', Georgia, 'Times New Roman', serif"

st.set_page_config(
    page_title="Data Guardians | Customer Segmentation Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Plotly theme - every chart inherits it
# --------------------------------------------------------------------------------------
pio.templates["guardians"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=13, color=INK),
        colorway=[GREEN, GOLD, SAGE, SLATE, DEEP, GOLD_LIGHT],
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title=dict(font=dict(family=SERIF, size=16, color=DEEP), x=0.0, xanchor="left", y=0.97),
        hoverlabel=dict(bgcolor=FOREST, bordercolor=FOREST, font=dict(family=FONT, size=13, color="#FFFFFF")),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, tickcolor=LINE,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED, size=12))),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED, size=12))),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=MUTED)),
        bargap=0.32,
    )
)
pio.templates.default = "plotly_white+guardians"

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Libre+Baskerville:wght@400;700&display=swap');

:root { --green:__GREEN__; --deep:__DEEP__; --forest:__FOREST__; --gold:__GOLD__; --gold-l:__GOLD_LIGHT__;
        --ink:__INK__; --muted:__MUTED__; --line:__LINE__; --mint:__MINT__; }

html, body, .stApp, [class*="css"] { font-family: __FONT__; color: var(--ink); }
.stApp {
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(201,162,39,.10), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(31,93,58,.08), transparent 60%),
    __PAGE__;
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4, h5 { font-family: __SERIF__; color: var(--deep); }

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"] {
  background: linear-gradient(185deg, var(--forest) 0%, var(--deep) 45%, var(--green) 100%);
  border-right: 0;
}
[data-testid="stSidebar"]::after { content:""; position:absolute; top:0; right:0; width:3px; height:100%;
  background: linear-gradient(180deg, var(--gold), rgba(201,162,39,0)); }
[data-testid="stSidebar"] h3 { color: var(--gold-l) !important; font-family: __FONT__; font-size: 12px !important;
  letter-spacing: 2.2px; text-transform: uppercase; font-weight: 700; margin: 18px 0 6px 0; padding: 0; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #E3EEE6 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] strong { color: var(--gold-l); }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,.06); border: 1.5px dashed rgba(235,203,107,.55); border-radius: 14px; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #E3EEE6 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
  background: var(--gold); border: 0; color: var(--forest) !important; font-weight: 600; border-radius: 10px; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * { color: var(--forest) !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * { color: #E3EEE6 !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.18); border-radius: 12px; }
[data-testid="stSidebar"] div[data-baseweb="select"] svg { color: #E3EEE6; fill: #E3EEE6; }
[data-testid="stSidebar"] span[data-baseweb="tag"] { background: var(--gold) !important; border-radius: 8px; }
[data-testid="stSidebar"] span[data-baseweb="tag"] span, [data-testid="stSidebar"] span[data-baseweb="tag"] svg {
  color: var(--forest) !important; fill: var(--forest) !important; font-weight: 600; }
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] { background: var(--gold-l) !important;
  box-shadow: 0 0 0 4px rgba(235,203,107,.25) !important; }
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] { color: var(--gold-l) !important; }
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div { background: rgba(255,255,255,.18); }
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div > div { background: var(--gold) !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label div:first-child { border-color: var(--gold-l) !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div { background: var(--gold) !important; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg { stroke: #E3EEE6; color: #E3EEE6; }
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stSidebarHeader"] svg { color: #E3EEE6; }
.side-brand { display:flex; align-items:center; gap:12px; padding: 4px 0 14px 0; border-bottom: 1px solid rgba(255,255,255,.12); }
.side-brand img { width: 58px; background:#FDFDFB; border-radius: 12px; padding: 4px; }
.side-brand .n { font-family: __SERIF__; color:#FDFDFB; font-size: 16px; line-height: 1.2; }
.side-brand .s { color: var(--gold-l); font-size: 11px; letter-spacing: 1.6px; text-transform: uppercase; font-weight: 600; }
.side-stats { display:grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }
.side-stats div { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 12px; padding: 9px 11px; }
.side-stats b { display:block; color:#FDFDFB; font-size: 16px; font-weight: 700; }
.side-stats span { color: #BFD3C6; font-size: 11px; letter-spacing: .6px; text-transform: uppercase; }
.side-stats .w { grid-column: 1 / span 2; }

/* ---------------- Hero ---------------- */
.hero { position: relative; overflow: hidden; border-radius: 26px; padding: 34px 40px 36px 40px; margin-bottom: 18px;
  background:
    radial-gradient(600px 300px at 100% 0%, rgba(235,203,107,.30), transparent 60%),
    radial-gradient(500px 300px at 0% 120%, rgba(111,179,139,.30), transparent 60%),
    linear-gradient(125deg, var(--forest) 0%, var(--deep) 35%, var(--green) 100%);
  box-shadow: 0 24px 50px -18px rgba(14,47,30,.55); display:grid; grid-template-columns: auto 1fr auto; gap: 32px; align-items:center; }
.hero::before { content:""; position:absolute; inset:0; opacity:.07;
  background-image: radial-gradient(#fff 1px, transparent 1px); background-size: 18px 18px; }
.hero::after { content:""; position:absolute; left:0; right:0; bottom:0; height:5px;
  background: linear-gradient(90deg, var(--gold), var(--gold-l), var(--gold)); }
.hero > * { position: relative; z-index: 1; }
.hero .logo { width: 112px; height: 112px; background:#FDFDFB; border-radius: 22px; display:flex; align-items:center; justify-content:center;
  box-shadow: 0 0 0 6px rgba(255,255,255,.08), 0 12px 24px rgba(0,0,0,.25); }
.hero .logo img { width: 94px; }
.hero .eyebrow { color: var(--gold-l); font-size: 12px; letter-spacing: 2.6px; font-weight: 700; text-transform: uppercase; }
.hero h1 { font-family: __SERIF__ !important; font-weight: 700 !important; color: #FDFDFB; font-size: 36px; line-height: 1.12; margin: 8px 0 10px 0; padding: 0; letter-spacing: -.3px; }
.hero h1 em { font-style: normal; color: var(--gold-l); }
.hero p { color: #D5E7DB; margin: 0; font-size: 15.5px; max-width: 780px; line-height: 1.55; }
.hero .pill { display:inline-flex; align-items:center; gap:7px; margin: 16px 8px 0 0; padding: 6px 14px; border-radius: 999px; font-size: 12.5px;
  background: rgba(255,255,255,.10); color: #FDFDFB; border: 1px solid rgba(255,255,255,.20); backdrop-filter: blur(6px); font-weight: 500; }
.hero .pill i { width:7px; height:7px; border-radius:50%; background: var(--gold-l); display:inline-block; }
.hero .facts { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); border-radius: 18px; padding: 16px 20px;
  backdrop-filter: blur(8px); min-width: 220px; }
.hero .facts .r { display:flex; justify-content: space-between; gap: 18px; padding: 7px 0; border-bottom: 1px dashed rgba(255,255,255,.16); }
.hero .facts .r:last-child { border-bottom: 0; }
.hero .facts span { color: #BFD3C6; font-size: 12px; letter-spacing: .8px; text-transform: uppercase; }
.hero .facts b { color: #FDFDFB; font-size: 14px; font-weight: 600; }
.live { display:inline-block; width:8px; height:8px; border-radius:50%; background:#7CE0A3; margin-right:6px;
  box-shadow: 0 0 0 0 rgba(124,224,163,.7); animation: pulse 2s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(124,224,163,.6);} 70% { box-shadow: 0 0 0 9px rgba(124,224,163,0);} 100% { box-shadow: 0 0 0 0 rgba(124,224,163,0);} }

.demo-banner { display:flex; gap: 14px; align-items:center; background: linear-gradient(90deg, #FFF7DD, #FFFBEF);
  border: 1px solid rgba(201,162,39,.45); color: #5B4A00; padding: 12px 18px; border-radius: 14px; margin-bottom: 16px; font-size: 14px; }
.demo-banner .ic { flex: 0 0 34px; height:34px; border-radius: 10px; background: var(--gold); color: var(--forest);
  display:flex; align-items:center; justify-content:center; }
.demo-banner code { background: rgba(201,162,39,.18); color: #5B4A00; }

/* ---------------- KPI cards ---------------- */
.kpi { position: relative; background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 16px 18px 14px 18px;
  min-height: 152px; box-shadow: 0 1px 2px rgba(14,47,30,.04), 0 8px 24px -12px rgba(14,47,30,.18);
  transition: transform .18s ease, box-shadow .18s ease; overflow: hidden; }
.kpi::before { content:""; position:absolute; left:18px; top:0; width:36px; height:3px; border-radius: 0 0 3px 3px; background: var(--accent, var(--green)); }
.kpi:hover { transform: translateY(-3px); box-shadow: 0 2px 4px rgba(14,47,30,.05), 0 18px 34px -14px rgba(14,47,30,.30); }
.kpi .top { display:flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
.kpi .l { font-size: 11.5px; letter-spacing: 1.3px; text-transform: uppercase; color: var(--muted); font-weight: 700; line-height: 1.35; padding-top: 4px; }
.kpi .ic { flex: 0 0 34px; height: 34px; border-radius: 11px; display:flex; align-items:center; justify-content:center;
  background: var(--accent-bg, var(--mint)); color: var(--accent, var(--green)); }
.kpi .v { font-family: __SERIF__; font-size: 29px; font-weight: 700; color: var(--deep); line-height: 1.15; margin-top: 10px; letter-spacing: -.5px; }
.kpi .s { font-size: 12.5px; color: var(--muted); margin-top: 4px; }
.kpi .bar { height: 5px; background: #EEF3EC; border-radius: 99px; margin-top: 10px; overflow: hidden; }
.kpi .bar i { display:block; height:100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent, var(--green)), var(--accent2, var(--sage, #6FB38B))); }

/* ---------------- Section headers ---------------- */
.sec { margin: 10px 0 14px 0; }
.sec .k { display:inline-flex; align-items:center; gap: 8px; color: var(--gold); font-size: 11.5px; font-weight: 700; letter-spacing: 2.2px; text-transform: uppercase; }
.sec .k::before { content:""; width: 22px; height: 2px; background: var(--gold); border-radius: 2px; }
.sec .t { font-family: __SERIF__; color: var(--deep); font-size: 25px; margin: 4px 0 4px 0; line-height: 1.25; }
.sec .s { color: var(--muted); font-size: 14.5px; }

/* ---------------- Bordered containers = cards ---------------- */
[class*="st-key-card"] { background: #fff !important; border: 1px solid var(--line) !important; border-radius: 20px !important;
  box-shadow: 0 1px 2px rgba(14,47,30,.04), 0 10px 28px -16px rgba(14,47,30,.22); padding: 6px 8px 2px 8px; }

.card { background:#fff; border: 1px solid var(--line); border-radius: 18px; padding: 18px 22px; margin-bottom: 14px;
  box-shadow: 0 1px 2px rgba(14,47,30,.04), 0 10px 28px -16px rgba(14,47,30,.20); }
.card h4 { margin: 0 0 6px 0; color: var(--deep); font-size: 17px; }
.card p { margin: 0; color: #2F3B34; font-size: 14.5px; line-height: 1.6; }
.stat-line { display:flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed var(--line); font-size: 14px; }
.stat-line:last-child { border-bottom: 0; }
.stat-line span { color: var(--muted); }
.stat-line b { color: var(--ink); }

/* Insight list */
.insight { display:flex; gap: 14px; align-items:flex-start; padding: 14px 4px; border-bottom: 1px solid #EEF3EC; }
.insight:last-child { border-bottom: 0; }
.insight .n { flex: 0 0 38px; height: 38px; border-radius: 12px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, var(--deep), var(--green)); color: var(--gold-l); font-family: __SERIF__; font-weight: 700; font-size: 16px; }
.insight .t { font-size: 11.5px; letter-spacing: 1.3px; text-transform: uppercase; color: var(--gold); font-weight: 700; }
.insight .b { font-size: 15px; color: var(--ink); margin-top: 3px; line-height: 1.5; }
.insight .b b { color: var(--deep); }

/* Cluster spotlight cards */
.cl { position: relative; background:#fff; border: 1px solid var(--line); border-radius: 20px; padding: 0; overflow: hidden; min-height: 248px;
  box-shadow: 0 1px 2px rgba(14,47,30,.04), 0 10px 28px -16px rgba(14,47,30,.22); transition: transform .18s ease, box-shadow .18s ease; }
.cl:hover { transform: translateY(-4px); box-shadow: 0 2px 4px rgba(14,47,30,.05), 0 22px 40px -18px rgba(14,47,30,.35); }
.cl .hd { padding: 16px 18px 14px 18px; color:#fff; position: relative; }
.cl .hd::after { content:""; position:absolute; right:-30px; top:-40px; width:130px; height:130px; border-radius:50%; background: rgba(255,255,255,.10); }
.cl .hd .row { display:flex; align-items:center; justify-content: space-between; }
.cl .hd .nm { font-family: __SERIF__; font-size: 19px; font-weight: 700; }
.cl .hd .ic { width: 34px; height: 34px; border-radius: 10px; background: rgba(255,255,255,.18); display:flex; align-items:center; justify-content:center; position: relative; z-index: 1; }
.cl .hd .gl { display:inline-block; margin-top: 6px; font-size: 11.5px; letter-spacing: 1.2px; text-transform: uppercase; font-weight: 700;
  background: rgba(255,255,255,.18); padding: 3px 10px; border-radius: 999px; }
.cl .bd { padding: 14px 18px 16px 18px; }
.cl .big { font-family: __SERIF__; font-size: 26px; color: var(--deep); font-weight: 700; }
.cl .big small { font-family: __FONT__; font-size: 12.5px; color: var(--muted); font-weight: 500; margin-left: 4px; }
.cl .meter { margin-top: 10px; }
.cl .meter .lb { display:flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.cl .meter .lb b { color: var(--ink); }
.cl .meter .tr { height: 7px; border-radius: 99px; background: #EEF3EC; overflow: hidden; }
.cl .meter .tr i { display:block; height: 100%; border-radius: 99px; }
.cl .ft { display:flex; gap: 8px; margin-top: 12px; }
.cl .ft div { flex:1; min-width:0; background: #F6F9F4; border-radius: 10px; padding: 7px 8px; }
.cl .ft span { display:block; font-size: 10px; color: var(--muted); letter-spacing: .3px; white-space: nowrap; text-transform: uppercase; font-weight: 600; }
.cl .ft b { font-size: 14px; color: var(--ink); }

/* Strategy hero card */
.strategy { position: relative; overflow: hidden; color: #FDFDFB; border-radius: 22px; padding: 24px 28px;
  background: radial-gradient(420px 200px at 100% 0%, rgba(235,203,107,.28), transparent 60%), linear-gradient(135deg, var(--forest), var(--green));
  box-shadow: 0 20px 40px -18px rgba(14,47,30,.55); }
.strategy::after { content:""; position:absolute; left:0; right:0; bottom:0; height:4px; background: linear-gradient(90deg, var(--gold), var(--gold-l)); }
.strategy .t { color: var(--gold-l); font-size: 12px; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; }
.strategy .a { font-family: __SERIF__; font-size: 24px; line-height: 1.35; margin: 10px 0 14px 0; }
.strategy .m { color: #D5E7DB; font-size: 14px; }
.chip { display:inline-block; background: var(--gold); color: var(--forest); padding: 3px 12px; border-radius: 999px; font-size: 12.5px;
  font-weight: 700; margin: 4px 6px 0 0; }
.chip.soft { background: var(--mint); color: var(--green); border: 1px solid #D3E3D6; font-weight: 600; }

/* Personalization matrix */
table.matrix { width: 100%; border-collapse: separate; border-spacing: 10px; table-layout: fixed; margin: 0 -10px; }
table.matrix th { background: transparent !important; border: 0 !important; color: var(--deep); font-size: 13px; letter-spacing: 1.2px; text-transform: uppercase; text-align: left; padding: 2px 6px; font-weight: 700; }
table.matrix th.row { background: #fff !important; border: 1px solid var(--line) !important; text-transform: none; letter-spacing: 0; font-family: __SERIF__; font-size: 16px; vertical-align: middle;
  background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 12px 14px; border-left: 6px solid var(--c) !important; }
table.matrix th.row span { display:block; font-family: __FONT__; color: var(--muted); font-weight: 600; font-size: 11.5px; letter-spacing: 1px;
  text-transform: uppercase; margin-top: 4px; }
table.matrix td { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 13px 15px; vertical-align: top;
  font-size: 13.5px; line-height: 1.45; color: #2F3B34; transition: all .15s ease; box-shadow: 0 6px 16px -12px rgba(14,47,30,.25); }
table.matrix td:hover { border-color: var(--sage, #6FB38B); transform: translateY(-2px); }
table.matrix td .cnt { display:block; font-size: 11.5px; color: var(--muted); margin-top: 8px; font-weight: 600; }
table.matrix td.sel { border: 2px solid var(--gold); background: linear-gradient(180deg, #FFFBEB, #FFF6D6);
  box-shadow: 0 14px 28px -14px rgba(201,162,39,.65); }

/* ---------------- Tabs as segmented control (works on old and new Streamlit tab markup) ---------------- */
[data-testid="stTabs"] [role="tablist"] { gap: 4px; background: #fff; padding: 6px; border-radius: 16px; border: 1px solid var(--line);
  box-shadow: 0 8px 22px -14px rgba(14,47,30,.25); width: fit-content; max-width: 100%; overflow-x: auto; border-bottom: 1px solid var(--line) !important; }
[data-testid="stTabs"] [role="tab"] { border-radius: 11px !important; padding: 9px 18px !important; height: auto !important; margin: 0 !important;
  background: transparent; transition: background .15s ease; border: 0 !important; cursor: pointer; }
[data-testid="stTabs"] [role="tab"] p { font-weight: 600; font-size: 14.5px; color: var(--muted); margin: 0; }
[data-testid="stTabs"] [role="tab"]:hover { background: var(--mint); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: linear-gradient(135deg, var(--deep), var(--green)) !important;
  box-shadow: 0 6px 14px -6px rgba(14,47,30,.6); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: #fff !important; }
[data-testid="stTabs"] .react-aria-SelectionIndicator, div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: 14px; }

/* Main-area radios as pills (old and new Streamlit radio markup) */
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] { gap: 6px; flex-wrap: wrap; }
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label {
  background: #fff; border: 1px solid var(--line); border-radius: 999px; padding: 6px 14px 6px 10px; margin: 0 !important;
  transition: all .15s ease; box-shadow: 0 4px 10px -8px rgba(14,47,30,.35); }
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label:hover { border-color: #6FB38B; }
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"],
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(135deg, var(--deep), var(--green)); border-color: var(--deep); }
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label[data-selected="true"] p,
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p { color: #fff !important; }

/* Inputs, tables, buttons */
[data-testid="stMain"] div[data-baseweb="select"] > div, [data-testid="stMain"] div[data-baseweb="input"] {
  border-radius: 12px !important; border-color: var(--line) !important; background: #fff; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 16px; overflow: hidden;
  box-shadow: 0 10px 28px -18px rgba(14,47,30,.25); background: #fff; }
.stDownloadButton button { background: linear-gradient(135deg, var(--deep), var(--green)); color: #fff; border: 0; border-radius: 12px;
  padding: 9px 20px; font-weight: 600; box-shadow: 0 8px 18px -8px rgba(14,47,30,.55); transition: transform .15s ease; }
.stDownloadButton button:hover { color: var(--gold-l); transform: translateY(-1px); border: 0; }
.stDownloadButton button p { color: inherit; }
h5 { font-size: 16px !important; margin-top: 6px; }
[data-testid="stAlert"] { border-radius: 14px; }

.footer { display:flex; justify-content: space-between; align-items:center; flex-wrap: wrap; gap: 10px; margin-top: 36px; padding: 18px 24px;
  border-radius: 18px; background: linear-gradient(125deg, var(--forest), var(--deep)); color: #BFD3C6; font-size: 13px; }
.footer b { color: #FDFDFB; font-family: __SERIF__; font-weight: 700; }
.footer span.g { color: var(--gold-l); }

@media (max-width: 1100px) {
  .hero { grid-template-columns: 1fr; padding: 26px; }
  .hero .facts { display: none; }
  .hero h1 { font-size: 28px; }
}
</style>
"""
_TOKENS = {
    "__GREEN__": GREEN, "__DEEP__": DEEP, "__FOREST__": FOREST, "__GOLD__": GOLD, "__GOLD_LIGHT__": GOLD_LIGHT,
    "__INK__": INK, "__MUTED__": MUTED, "__LINE__": LINE, "__MINT__": MINT, "__PAGE__": PAGE,
    "__FONT__": FONT, "__SERIF__": SERIF,
}
for _k, _v in _TOKENS.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Icons (simple inline SVG line icons)
# --------------------------------------------------------------------------------------
_ICON_PATHS = {
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "receipt": '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/>',
    "dollar": '<path d="M12 1v22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "pie": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "award": '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    "trend": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
             '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "moon": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
}
CLUSTER_ICONS = {"Champions": "award", "Loyal/Potential": "trend", "At-Risk": "alert", "Inactive": "moon"}


def icon(name: str, size: int = 18, stroke: float = 2.0) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')


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


def kpi(label: str, value: str, sub: str = "", ic: str = "users", accent: str = GREEN,
        accent_bg: str = MINT, pct: float | None = None) -> str:
    bar = ""
    if pct is not None:
        bar = f'<div class="bar"><i style="width:{max(0.0, min(pct, 100.0)):.1f}%"></i></div>'
    return (
        f'<div class="kpi" style="--accent:{accent};--accent-bg:{accent_bg}">'
        f'<div class="top"><div class="l">{esc(label)}</div><div class="ic">{icon(ic)}</div></div>'
        f'<div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div>{bar}</div>'
    )


def show(fig: go.Figure, key: str | None = None) -> None:
    config = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
    try:
        st.plotly_chart(fig, width="stretch", key=key, config=config, theme=None)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key, config=config, theme=None)


def style(fig: go.Figure, height: int = 400, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=8, r=8, t=64, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
    )
    return fig


def card(key: str):
    """A white rounded card around Streamlit content (charts, widgets)."""
    return st.container(border=True, key=f"card_{key}")


def section(kicker: str, title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="sec"><div class="k">{esc(kicker)}</div><div class="t">{esc(title)}</div>'
        f'<div class="s">{esc(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


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


logo = logo_b64()

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    brand_img = f'<img src="data:image/png;base64,{logo}" alt="FUSE logo">' if logo else ""
    st.markdown(
        f'<div class="side-brand">{brand_img}<div><div class="s">Team</div><div class="n">Data Guardians</div></div></div>',
        unsafe_allow_html=True,
    )
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
    st.caption(f"Source: **{source_label}**")

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
    st.markdown("### Dataset")
    st.markdown(
        f"""<div class="side-stats">
        <div><b>{len(customers):,}</b><span>Customers</span></div>
        <div><b>{len(result.transactions):,}</b><span>Order records</span></div>
        <div class="w"><b>{result.reference_date.strftime('%d %b %Y')}</b><span>Reference date</span></div>
        </div>""",
        unsafe_allow_html=True,
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
logo_html = f'<div class="logo"><img src="data:image/png;base64,{logo}" alt="FUSE logo"></div>' if logo else ""
st.markdown(
    f"""
<div class="hero">
  {logo_html}
  <div>
    <div class="eyebrow">Florida University Southeast · Capstone · BI &amp; Data Analytics</div>
    <h1>Customer Segmentation &amp; <em>Personalization</em> Engine</h1>
    <p>RFM analysis and K-Means clustering turn order records into four customer segments and twelve
       segment-specific actions for an online retail store.</p>
    <span class="pill"><i></i>Team Data Guardians</span><span class="pill"><i></i>Global Superstore data</span><span class="pill"><i></i>K = 4 · RFM 3–15</span>
  </div>
  <div class="facts">
    <div class="r"><span>Status</span><b><span class="live"></span>{'Demo' if is_demo else 'Live data'}</b></div>
    <div class="r"><span>Reference</span><b>{result.reference_date.strftime('%d %b %Y')}</b></div>
    <div class="r"><span>Dates</span><b>{'Rebuilt (weekly)' if result.date_source == 'rebuilt' else 'As recorded'}</b></div>
    <div class="r"><span>In view</span><b>{len(view):,} customers</b></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
if is_demo:
    st.markdown(
        f'<div class="demo-banner"><div class="ic">{icon("info")}</div><div><b>Demo data.</b> These customers are synthetic and only imitate the report. '
        "Upload the real Global Superstore file in the sidebar (or put it next to <code>app.py</code>) to see real results.</div></div>",
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
GOLD_BG = "#FBF3D6"
cols = st.columns(6)
cards = [
    ("Customers", f"{len(view):,}", f"{len(view) / len(customers) * 100:.0f}% of all accounts", "users", GREEN, MINT,
     len(view) / len(customers) * 100),
    ("Order records", f"{len(tx_view):,}", f"{len(tx_view) / max(len(view), 1):.1f} per customer", "receipt", GOLD, GOLD_BG,
     len(tx_view) / max(len(result.transactions), 1) * 100),
    ("Revenue", money(view["Monetary"].sum()), f"{money(view['Monetary'].mean())} per customer", "dollar", GREEN, MINT,
     view["Monetary"].sum() / customers["Monetary"].sum() * 100),
    ("Avg RFM score", f"{view['RFM'].mean():.1f}", "on a 3 to 15 scale", "star", GOLD, GOLD_BG,
     (view["RFM"].mean() - 3) / 12 * 100),
    ("Champions + At-Risk", f"{top2_share:.1f}%", f"of revenue from {top2_cust:.0f}% of customers", "pie", GREEN, MINT,
     top2_share),
    ("Silhouette", f"{result.silhouette:.2f}", "cluster separation (moderate)", "target", GOLD, GOLD_BG,
     max(result.silhouette, 0) * 100),
]
for col, (label, value, sub, ic, acc, acc_bg, pct) in zip(cols, cards):
    col.markdown(kpi(label, value, sub, ic, acc, acc_bg, pct), unsafe_allow_html=True)
st.write("")

tab_over, tab_seg, tab_acc, tab_pers, tab_cust, tab_model = st.tabs(
    ["Overview", "Segments", "Account types", "Personalization", "Customer explorer", "Model & data quality"]
)

# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------
with tab_over:
    section("01 · Overview", "Who are our customers, and where does the money come from?",
            "Four behavioral clusters, their size, and the revenue each one brings in.")

    # Cluster spotlight cards
    spot = st.columns(len(prof))
    for col, name in zip(spot, prof.index):
        c = CLUSTER_COLORS[name]
        p = prof.loc[name]
        col.markdown(
            f"""<div class="cl">
              <div class="hd" style="background:linear-gradient(135deg,{c},{rgba(c, .82)})">
                <div class="row"><div class="nm">{esc(name)}</div><div class="ic">{icon(CLUSTER_ICONS[name])}</div></div>
                <span class="gl">{esc(E.CLUSTER_META[name]['goal'])}</span>
              </div>
              <div class="bd">
                <div class="big">{int(p['Customers']):,}<small>customers</small></div>
                <div class="meter"><div class="lb"><span>Share of customers</span><b>{p['Pct customers']:.1f}%</b></div>
                  <div class="tr"><i style="width:{p['Pct customers']:.1f}%;background:{c}"></i></div></div>
                <div class="meter"><div class="lb"><span>Share of revenue</span><b>{p['Pct revenue']:.1f}%</b></div>
                  <div class="tr"><i style="width:{p['Pct revenue']:.1f}%;background:{GOLD}"></i></div></div>
                <div class="ft"><div><span>Recency</span><b>{p['Recency']:.0f} d</b></div>
                  <div><span>Avg spend</span><b>{money(p['Monetary'])}</b></div>
                  <div><span>RFM</span><b>{p['RFM']:.1f}</b></div></div>
              </div></div>""",
            unsafe_allow_html=True,
        )
    st.write("")

    c1, c2 = st.columns([1, 1.35])
    with c1, card("donut"):
        fig = go.Figure(
            go.Pie(
                labels=prof.index.tolist(),
                values=prof["Customers"],
                hole=0.68,
                sort=False,
                direction="clockwise",
                marker=dict(colors=[CLUSTER_COLORS[c] for c in prof.index], line=dict(color="#FFFFFF", width=4)),
                textinfo="percent",
                textfont=dict(color="#FFFFFF", size=13, family=FONT),
                hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>",
            )
        )
        fig.add_annotation(text=f"<b>{len(view):,}</b>", showarrow=False, y=0.53, yref="paper",
                           font=dict(family=SERIF, size=34, color=DEEP))
        fig.add_annotation(text="CUSTOMERS", showarrow=False, y=0.43, yref="paper", font=dict(size=11, color=MUTED))
        style(fig, 420).update_layout(
            title_text="Customers by cluster",
            legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
            margin=dict(l=8, r=8, t=56, b=40),
        )
        show(fig, "donut")
    with c2, card("dumbbell"):
        order = prof.index.tolist()[::-1]
        fig = go.Figure()
        for name in order:
            a, b = prof.loc[name, "Pct customers"], prof.loc[name, "Pct revenue"]
            fig.add_trace(go.Scatter(x=[a, b], y=[name, name], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=rgba(CLUSTER_COLORS[name], .35), width=10)))
        fig.add_trace(go.Scatter(
            x=prof.loc[order, "Pct customers"], y=order, mode="markers+text", name="% of customers",
            marker=dict(size=20, color="#FFFFFF", line=dict(color=GREEN, width=4)),
            text=[f"{v:.1f}%" for v in prof.loc[order, "Pct customers"]], textposition="bottom center",
            textfont=dict(color=GREEN, size=12), hovertemplate="%{y}<br>%{x:.1f}% of customers<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=prof.loc[order, "Pct revenue"], y=order, mode="markers+text", name="% of revenue",
            marker=dict(size=22, color=GOLD, line=dict(color="#FFFFFF", width=3), symbol="diamond"),
            text=[f"{v:.1f}%" for v in prof.loc[order, "Pct revenue"]], textposition="top center",
            textfont=dict(color="#8A6D10", size=12), hovertemplate="%{y}<br>%{x:.1f}% of revenue<extra></extra>",
        ))
        hi = max(prof["Pct customers"].max(), prof["Pct revenue"].max())
        style(fig, 420).update_layout(
            title_text="Share of customers vs share of revenue",
            xaxis=dict(title="% of total", range=[0, hi * 1.15], ticksuffix="%"),
            yaxis=dict(showgrid=False, tickfont=dict(size=13, color=INK)),
        )
        show(fig, "dumbbell")

    c3, c4 = st.columns([1.35, 1])
    with c3, card("treemap"):
        tm = view.groupby(["Cluster", "Segment"], observed=True)["Monetary"].sum().reset_index()
        tm["Cluster"] = tm["Cluster"].astype(str)
        fig = px.treemap(tm, path=[px.Constant("All revenue"), "Cluster", "Segment"], values="Monetary",
                         color="Cluster", color_discrete_map={**CLUSTER_COLORS, "(?)": DEEP})
        fig.update_traces(
            root_color=FOREST, marker=dict(line=dict(color="#FFFFFF", width=3), cornerradius=8),
            texttemplate="<b>%{label}</b><br>%{value:$,.0f}<br>%{percentRoot:.1%}", textposition="middle center",
            textfont=dict(family=FONT, color="#FFFFFF", size=15), pathbar=dict(visible=False),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f} revenue<br>%{percentRoot:.1%} of total<extra></extra>",
        )
        style(fig, 440, legend=False).update_layout(title_text="Revenue by cluster and account type",
                                                    margin=dict(l=8, r=8, t=56, b=8), uniformtext=dict(minsize=11, mode="hide"))
        show(fig, "treemap")
    with c4, card("insights"):
        st.markdown('<div class="sec" style="margin:8px 6px 0 6px"><div class="k">Insights</div>'
                    '<div class="t" style="font-size:20px">What the data says</div></div>', unsafe_allow_html=True)
        biggest = prof["Customers"].idxmax()
        items = [
            ("Largest group", f"<b>{esc(biggest)}</b> is the largest cluster: {prof.loc[biggest, 'Pct customers']:.1f}% of customers "
                              f"but {prof.loc[biggest, 'Pct revenue']:.1f}% of revenue."),
            ("Revenue concentration", f"<b>Champions + At-Risk</b> are {top2_cust:.1f}% of customers and "
                                      f"<b>{top2_share:.1f}%</b> of revenue."),
        ]
        if {"Champions", "At-Risk"} <= set(prof.index):
            ratio = prof.loc["At-Risk", "Recency"] / prof.loc["Champions", "Recency"]
            items.append(("Retention signal", f"<b>At-Risk</b> customers have been silent <b>{ratio:.1f}×</b> longer than Champions "
                                              f"({prof.loc['At-Risk', 'Recency']:.0f} vs {prof.loc['Champions', 'Recency']:.0f} days). "
                                              "Re-engage them first."))
        else:
            items.append(("Retention signal", "Select both Champions and At-Risk to compare how long they have been silent."))
        st.markdown(
            '<div style="padding:0 6px 8px 6px">' + "".join(
                f'<div class="insight"><div class="n">{i}</div><div><div class="t">{esc(t)}</div><div class="b">{b}</div></div></div>'
                for i, (t, b) in enumerate(items, 1)
            ) + "</div>",
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------------------
with tab_seg:
    section("02 · Segments", "Cluster profiles",
            "Mean behavior of each cluster. Revenue share = cluster revenue divided by total revenue.")
    table = prof.reset_index()[["Cluster", "Customers", "Pct customers", "Recency", "Frequency", "Monetary", "RFM", "Pct revenue"]]
    table["Cluster"] = table["Cluster"].astype(str)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
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
    st.write("")
    c1, c2 = st.columns([1, 1.25])
    with c1, card("radar"):
        theta = ["Recency", "Frequency", "Monetary", "Recency"]
        fig = go.Figure()
        for name in prof.index:
            r = [prof.loc[name, "R"], prof.loc[name, "F"], prof.loc[name, "M"], prof.loc[name, "R"]]
            fig.add_trace(
                go.Scatterpolar(r=r, theta=theta, name=name, fill="toself", fillcolor=rgba(CLUSTER_COLORS[name], .16),
                                line=dict(color=CLUSTER_COLORS[name], width=3), marker=dict(size=7),
                                hovertemplate="%{theta} score: %{r:.2f}<extra>" + name + "</extra>")
            )
        style(fig, 500).update_layout(
            title_text="Average RFM scores (1 to 5)",
            polar=dict(
                radialaxis=dict(range=[0, 5], gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED, size=10), angle=90),
                angularaxis=dict(gridcolor=GRID, linecolor=LINE, tickfont=dict(size=13, color=INK)),
                bgcolor="rgba(0,0,0,0)",
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5),
            margin=dict(l=60, r=60, t=70, b=50),
        )
        show(fig, "radar")
    with c2, card("scatter"):
        axes = {"Recency (days)": "Recency", "Frequency (records)": "Frequency", "Monetary ($)": "Monetary"}
        ca, cb, cc = st.columns([1, 1, 0.8])
        xlab = ca.selectbox("X axis", list(axes), index=0)
        ylab = cb.selectbox("Y axis", list(axes), index=2)
        cc.write("")
        logy = cc.toggle("Log scale on Y", value=True)
        sample = view.sample(min(len(view), 3000), random_state=1)
        fig = px.scatter(
            sample, x=axes[xlab], y=axes[ylab], color=sample["Cluster"].astype(str), size="Frequency",
            size_max=15, color_discrete_map=CLUSTER_COLORS, opacity=0.7,
            hover_name="Customer Name", hover_data={"Customer ID": True, "Segment": True, "RFM": True},
            category_orders={"color": E.CLUSTER_ORDER}, log_y=logy,
        )
        fig.update_traces(marker=dict(line=dict(color="#FFFFFF", width=0.6)))
        style(fig, 410).update_layout(title_text=f"{ylab} against {xlab}", legend_title_text="")
        fig.update_xaxes(title=xlab)
        fig.update_yaxes(title=ylab, **({"dtick": 1} if logy else {}))
        show(fig, "scatter")
    with card("box"):
        metric = st.radio("Distribution of", list(axes), horizontal=True)
        bx = view.assign(ClusterName=view["Cluster"].astype(str))
        fig = go.Figure()
        for name in [c for c in E.CLUSTER_ORDER if c in set(bx["ClusterName"])]:
            fig.add_trace(go.Box(
                y=bx.loc[bx["ClusterName"] == name, axes[metric]], name=name, boxpoints=False,
                marker_color=CLUSTER_COLORS[name], fillcolor=rgba(CLUSTER_COLORS[name], .22),
                line=dict(color=CLUSTER_COLORS[name], width=2), boxmean=True,
            ))
        if axes[metric] != "Recency":
            fig.update_yaxes(type="log", dtick=1)
        style(fig, 380, legend=False).update_layout(title_text=f"{metric} by cluster", yaxis_title=metric)
        show(fig, "box")

# --------------------------------------------------------------------------------------
# Account types
# --------------------------------------------------------------------------------------
with tab_acc:
    section("03 · Account types", "Does behavior depend on account type?",
            "Cross-tabulation of behavioral cluster against Consumer, Corporate and Home Office accounts.")
    cross = pd.crosstab(view["Cluster"], view["Segment"]).reindex(E.CLUSTER_ORDER).dropna(how="all")
    cross = cross.reindex(columns=[c for c in E.SEGMENT_ORDER if c in cross.columns])
    pct_row = cross.div(cross.sum(axis=1), axis=0) * 100
    c1, c2 = st.columns(2)
    with c1, card("heat"):
        fig = go.Figure(go.Heatmap(
            z=pct_row.values, x=list(pct_row.columns), y=[str(i) for i in pct_row.index],
            colorscale=[[0, "#F1F6EF"], [0.5, SAGE], [1, DEEP]], showscale=False, xgap=6, ygap=6,
            text=pct_row.values, texttemplate="<b>%{text:.1f}%</b>", textfont=dict(size=15, family=FONT),
            hovertemplate="%{y} · %{x}<br>%{z:.1f}% of the cluster<extra></extra>",
        ))
        style(fig, 400, legend=False).update_layout(title_text="Account-type mix inside each cluster (%)",
                                                    yaxis=dict(autorange="reversed", showgrid=False),
                                                    xaxis=dict(showgrid=False, side="top"))
        show(fig, "heat")
    with c2, card("stack"):
        long = cross.reset_index().melt(id_vars="Cluster", var_name="Account type", value_name="Customers")
        long["Cluster"] = long["Cluster"].astype(str)
        fig = px.bar(
            long, x="Customers", y="Cluster", color="Account type", orientation="h", barmode="stack",
            color_discrete_map=ACCOUNT_COLORS, text="Customers",
            category_orders={"Cluster": E.CLUSTER_ORDER[::-1], "Account type": E.SEGMENT_ORDER},
        )
        fig.update_traces(textfont=dict(color="#FFFFFF", size=12), marker_line_color="#FFFFFF", marker_line_width=2,
                          hovertemplate="%{y} · %{fullData.name}<br>%{x:,} customers<extra></extra>")
        style(fig, 400).update_layout(title_text="Customers by cluster and account type", yaxis_title="", bargap=0.35)
        show(fig, "stack")
    chi = E.chi_square(customers)
    independent = chi["p"] > 0.05
    verdict = (
        "No evidence that cluster membership depends on account type. Each cluster has roughly the same Consumer, "
        "Corporate and Home Office mix, and the effect size is negligible."
        if independent
        else "Cluster membership and account type are statistically related in this data; check the mix above."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Chi-square", f"{chi['chi2']:.2f}", f"df = {chi['dof']}, N = {chi['n']:,}", "pie"), unsafe_allow_html=True)
    k2.markdown(kpi("p-value", f"{chi['p']:.3f}", "above 0.05 = independent" if independent else "below 0.05 = related",
                    "target", GOLD, GOLD_BG), unsafe_allow_html=True)
    k3.markdown(kpi("Cramér's V", f"{chi['v']:.2f}", "effect size (0 = none)", "star"), unsafe_allow_html=True)
    k4.markdown(kpi("Verdict", "Independent" if independent else "Related", "cluster vs account type", "info", GOLD, GOLD_BG),
                unsafe_allow_html=True)
    st.markdown(
        f"""<div class="card" style="margin-top:14px"><h4>Chi-square test of independence (all customers)</h4>
        <p><b>χ²({chi['dof']}, N = {chi['n']:,}) = {chi['chi2']:.2f}, p = {chi['p']:.3f}, Cramér's V = {chi['v']:.2f}.</b> {esc(verdict)}</p>
        <p style="margin-top:8px"><b>Use both dimensions:</b> behavior (the cluster) decides the goal and urgency; account type decides the channel and offer.</p></div>""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# Personalization
# --------------------------------------------------------------------------------------
with tab_pers:
    section("04 · Personalization", "The personalization matrix",
            "4 clusters × 3 account types = 12 strategies. Pick a cell to see its action and leading product sub-categories.")
    pc1, pc2 = st.columns(2)
    sel_c = pc1.radio("Cluster", E.CLUSTER_ORDER, horizontal=True, key="pc")
    sel_s = pc2.radio("Account type", E.SEGMENT_ORDER, horizontal=True, key="ps")
    focus = result.focus
    top2: dict[tuple[str, str], list[str]] = {}
    if focus is not None:
        for (cl, ac), grp in focus[focus["Rank"] <= 2].groupby(["Cluster", "Account"], observed=True):
            top2[(str(cl), str(ac))] = grp.sort_values("Rank")["Sub-Category"].tolist()

    cell_counts = view.groupby(["Cluster", "Segment"], observed=True).size()
    head = "".join(f"<th>{esc(s)}</th>" for s in E.SEGMENT_ORDER)
    body = ""
    for cl in E.CLUSTER_ORDER:
        body += (f'<tr><th class="row" style="--c:{CLUSTER_COLORS[cl]}">{esc(cl)}'
                 f'<span>{esc(E.CLUSTER_META[cl]["goal"])}</span></th>')
        for ac in E.SEGMENT_ORDER:
            chips = "".join(f'<span class="chip soft">{esc(s)}</span>' for s in top2.get((cl, ac), []))
            cls = ' class="sel"' if (cl, ac) == (sel_c, sel_s) else ""
            n = int(cell_counts.get((cl, ac), 0))
            body += f'<td{cls}>{esc(E.STRATEGIES[(cl, ac)])}<div>{chips}</div><span class="cnt">{n:,} customers in view</span></td>'
        body += "</tr>"
    st.markdown(f'<table class="matrix"><tr><th style="width:17%"></th>{head}</tr>{body}</table>', unsafe_allow_html=True)

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
        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.markdown(kpi("Customers in cell", f"{len(cell_cust):,}", "after filters", "users"), unsafe_allow_html=True)
        m2.markdown(kpi("Revenue", money(cell_cust["Monetary"].sum()) if len(cell_cust) else "$0", "total sales", "dollar",
                        GOLD, GOLD_BG), unsafe_allow_html=True)
        m3.markdown(kpi("Avg recency", f"{cell_cust['Recency'].mean():.0f} d" if len(cell_cust) else "-",
                        "days since last order", "clock"), unsafe_allow_html=True)
    with right, card("focus"):
        if focus is not None:
            cell_focus = focus[(focus["Cluster"] == sel_c) & (focus["Account"] == sel_s)].sort_values("Rank").head(8)
            cell_focus = cell_focus.assign(Top=np.where(cell_focus["Rank"] <= 2, "Top two", "Other"))
            fig = px.bar(
                cell_focus.iloc[::-1], x="Share", y="Sub-Category", color="Top", orientation="h",
                color_discrete_map={"Top two": GOLD, "Other": "#BFD3C6"}, text="Share",
            )
            fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", textfont=dict(color=MUTED, size=12),
                              hovertemplate="%{y}: %{x:.1f}% of sales<extra></extra>", marker_line_width=0)
            style(fig, 400).update_layout(title_text="Sub-categories by share of the cell's sales",
                                          xaxis_title="% of cell sales", yaxis_title="", bargap=0.3,
                                          xaxis=dict(range=[0, cell_focus["Share"].max() * 1.25 if len(cell_focus) else 1]))
            show(fig, "focus")
        else:
            st.info("This file has no Sub-Category column, so product focus is unavailable.")

    st.write("")
    section("Action plan", "Recommended priorities", "Where to act first, and how to know it worked.")
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
        pd.DataFrame(prio), hide_index=True, width="stretch",
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
    st.download_button("⬇  Download the 12-strategy matrix (CSV)", pd.DataFrame(matrix_rows).to_csv(index=False).encode(),
                       file_name="personalization_matrix.csv", mime="text/csv")

# --------------------------------------------------------------------------------------
# Customer explorer
# --------------------------------------------------------------------------------------
with tab_cust:
    section("05 · Customer explorer", "Find a customer and see the recommended action",
            "Search by Customer ID or name. The table follows the sidebar filters.")
    q = st.text_input("Search", placeholder="🔍  Type part of a Customer ID or name")
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
        shown, hide_index=True, height=360, width="stretch",
        column_config={
            "Segment": "Account type",
            "Recency": st.column_config.NumberColumn("Recency (days)", format="%d"),
            "Frequency": st.column_config.NumberColumn("Frequency", format="%d"),
            "Monetary": st.column_config.NumberColumn("Monetary ($)", format="$%.0f"),
            "RFM": st.column_config.ProgressColumn("RFM total", min_value=3, max_value=15, format="%d"),
            "Action": st.column_config.TextColumn("Recommended action", width="large"),
        },
    )
    st.download_button("⬇  Download these customers (CSV)", shown.to_csv(index=False).encode(),
                       file_name="customer_segments.csv", mime="text/csv")

    if len(found):
        st.write("")
        section("Customer 360", "One customer, one clear next step")
        options = found.sort_values("Monetary", ascending=False).head(500)
        labels = {r["Customer ID"]: f'{r["Customer Name"]}  ({r["Customer ID"]})' for _, r in options.iterrows()}
        chosen = st.selectbox("Choose a customer", list(labels), format_func=lambda k: labels[k])
        row = customers[customers["Customer ID"] == chosen].iloc[0]
        cluster = str(row["Cluster"])
        ccol = CLUSTER_COLORS[cluster]
        g1, g2, g3 = st.columns([1, 1.15, 1.4])
        with g1, card("gauge"):
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number", value=float(row["RFM"]),
                    number=dict(font=dict(color=DEEP, size=46, family=SERIF)),
                    gauge=dict(
                        axis=dict(range=[3, 15], tickcolor=MUTED, tickfont=dict(color=MUTED, size=11), dtick=3),
                        bar=dict(color=ccol, thickness=0.3),
                        bgcolor="#F1F6EF",
                        steps=[dict(range=[3, 7], color="#F1F6EF"), dict(range=[7, 11], color="#E4EEE2"), dict(range=[11, 15], color="#D3E5D7")],
                        threshold=dict(line=dict(color=GOLD, width=4), thickness=0.85, value=float(customers["RFM"].mean())),
                        borderwidth=0,
                    ),
                    title=dict(text="RFM total · gold line = average", font=dict(size=13, color=MUTED)),
                )
            )
            style(fig, 270, legend=False).update_layout(margin=dict(l=24, r=24, t=50, b=8))
            show(fig, "gauge")
        with g2:
            st.markdown(
                f"""<div class="card" style="border-top:5px solid {ccol}"><h4>{esc(row['Customer Name'])}</h4>
                <p style="color:{MUTED};font-size:13px">{esc(row['Customer ID'])} · {esc(row['Segment'])}
                <span class="chip" style="background:{ccol};color:#fff;margin-left:6px">{esc(cluster)}</span></p>
                <div style="margin-top:10px">
                  <div class="stat-line"><span>Last order</span><b>{row['Recency']:,} days ago</b></div>
                  <div class="stat-line"><span>Order records</span><b>{int(row['Frequency']):,}</b></div>
                  <div class="stat-line"><span>Total sales</span><b>${row['Monetary']:,.0f}</b></div>
                  <div class="stat-line"><span>Scores R / F / M</span><b>{int(row['R'])} / {int(row['F'])} / {int(row['M'])}</b></div>
                </div></div>""",
                unsafe_allow_html=True,
            )
        with g3:
            tops = top2.get((cluster, str(row["Segment"])), [])
            chips = "".join(f'<span class="chip">{esc(s)}</span>' for s in tops)
            st.markdown(
                f"""<div class="strategy"><div class="t">Recommended action · {esc(E.CLUSTER_META[cluster]['goal'])}</div>
                <div class="a" style="font-size:21px">{esc(row['Action'])}</div>
                <div class="m"><b>Lead with:</b> {chips if chips else 'n/a'}</div>
                <div class="m" style="margin-top:10px"><b>Measure:</b> {esc(E.CLUSTER_META[cluster]['measures'])}</div></div>""",
                unsafe_allow_html=True,
            )
        history = result.transactions[result.transactions["Customer ID"] == chosen]
        hist = history.groupby("OrderDate")["Sales"].sum().reset_index()
        with card("history"):
            fig = go.Figure(go.Bar(x=hist["OrderDate"], y=hist["Sales"], marker=dict(color=ccol, cornerradius=4),
                                   hovertemplate="Week of %{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"))
            style(fig, 260, legend=False).update_layout(
                title_text="Sales by recorded order week (dates are the Monday of each week)", yaxis_title="Sales ($)")
            show(fig, "history")

# --------------------------------------------------------------------------------------
# Model and data quality
# --------------------------------------------------------------------------------------
with tab_model:
    section("06 · Model & data quality", "How good is the model?",
            "Elbow curve, silhouette by K, stability across random seeds, and a check against the report.")
    diag = diagnostics(result.scaled, customers["Label"].to_numpy())
    c1, c2 = st.columns(2)
    with c1, card("elbow"):
        elbow = diag["elbow"]
        fig = go.Figure()
        fig.add_scatter(x=elbow["K"], y=elbow["Inertia"], mode="lines+markers", line=dict(color=GREEN, width=3, shape="spline"),
                        fill="tozeroy", fillcolor=rgba(GREEN, .08),
                        marker=dict(size=9, color="#FFFFFF", line=dict(color=GREEN, width=3)), name="Inertia")
        sel = elbow[elbow["K"] == E.K_CLUSTERS]
        fig.add_scatter(x=sel["K"], y=sel["Inertia"], mode="markers", marker=dict(size=20, color=GOLD, line=dict(color="#FFFFFF", width=3)),
                        name="Selected K = 4")
        if len(sel):
            fig.add_annotation(x=int(sel["K"].iloc[0]), y=float(sel["Inertia"].iloc[0]), text="<b>K = 4</b>", showarrow=True,
                               arrowhead=0, ax=40, ay=-40, arrowcolor=GOLD, font=dict(color="#8A6D10"),
                               bgcolor="#FFFBEB", bordercolor=GOLD, borderpad=4)
        style(fig, 370).update_layout(title_text="Elbow curve (inertia)", xaxis=dict(title="Number of clusters K", dtick=1),
                                      yaxis_title="Inertia")
        show(fig, "elbow")
    with c2, card("sil"):
        sil = diag["silhouette"]
        fig = go.Figure(go.Bar(
            x=sil["K"], y=sil["Silhouette"], text=[f"{v:.2f}" for v in sil["Silhouette"]], textposition="outside",
            marker=dict(color=[GOLD if k == E.K_CLUSTERS else "#BFD3C6" for k in sil["K"]], cornerradius=6),
            textfont=dict(color=MUTED), hovertemplate="K = %{x}<br>Silhouette %{y:.3f}<extra></extra>",
        ))
        style(fig, 370, legend=False).update_layout(title_text="Silhouette score by K", xaxis=dict(dtick=1, title="K"),
                                                    yaxis=dict(range=[0, sil["Silhouette"].max() * 1.2]))
        show(fig, "sil")
    st.caption("The elbow falls steeply up to K = 4 and then declines gradually with no sharp plateau, so K = 4 rests on the curve, "
               "the four marketing goals it supports, and stability. Choosing K is a judgment.")

    s1, s2 = st.columns([1, 1.4])
    with s1:
        stab = diag["stability"].copy()
        st.markdown("##### Stability across random seeds")
        st.dataframe(stab, hide_index=True, width="stretch",
                     column_config={"Adjusted Rand index": st.column_config.ProgressColumn("Adjusted Rand index", min_value=0, max_value=1, format="%.3f")})
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
        st.dataframe(check, hide_index=True, width="stretch")
        if is_demo:
            st.caption("Demo data is synthetic, so differences from the report are expected.")
        else:
            st.caption("Small differences can come from the quintile tie-breaking rule or the date source; see the README.")

    st.write("")
    section("Data quality", "Why the report rebuilds the Order Date")
    info = result.date_info
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(kpi("Order records", f"{len(result.transactions):,}", f"{raw_df.shape[1]} columns", "receipt"), unsafe_allow_html=True)
    if info.get("has_order_date"):
        d2.markdown(kpi("Distinct Order Dates", f"{info.get('distinct_original', 0):,}", "as-is · 1 means corrupted", "alert",
                        GOLD, GOLD_BG), unsafe_allow_html=True)
        d3.markdown(kpi("Recency SD (as-is)", f"{info.get('recency_sd_original', float('nan')):.1f}",
                        "0.0 = customers indistinguishable", "clock", GOLD, GOLD_BG), unsafe_allow_html=True)
    rec = customers["Recency"]
    d4.markdown(kpi("Recency SD (used)", f"{rec.std(ddof=1):.1f}", f"mean {rec.mean():.1f}, range {rec.min()}–{rec.max()} d", "clock"),
                unsafe_allow_html=True)
    st.markdown(
        f"""<div class="card" style="margin-top:14px"><h4>Method in one paragraph</h4><p>
        Dates are rebuilt as the Monday of each recorded week from Year and week number, so a date can be off by up to 6 days.
        The reference date is {result.reference_date.strftime('%d %B %Y')} (one day after the latest order week). Recency is days since the last order
        (scored in reverse), Frequency counts order records (product lines), and Monetary is total Sales; each is scored 1 to 5 by quintile.
        The three scores are standardized and clustered with K-Means (K = 4, random_state = 42, n_init = 10).</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("##### Limitations to state openly")
    lim = st.columns(3)
    limits = [
        ("clock", "Weekly dates", "Rebuilt dates can be off by up to 6 days."),
        ("target", "Moderate separation", "Silhouette near 0.38: the clusters overlap at their edges."),
        ("star", "Quintile scores", "Scores keep rank order but not the size of differences."),
        ("users", "Customer ID", "One ID is treated as one account; some names appear under several IDs."),
        ("receipt", "Historical data", "The data cover 2011 to 2014: the method transfers, the numbers do not."),
        ("alert", "No campaign test", "The strategies are proposed rules; their effect has not been measured."),
    ]
    for i, (ic, title, text) in enumerate(limits):
        lim[i % 3].markdown(
            f'<div class="card" style="display:flex;gap:14px;align-items:flex-start">'
            f'<div class="insight" style="padding:0;border:0"><div class="n">{icon(ic, 17)}</div></div>'
            f'<div><h4 style="font-size:16px">{esc(title)}</h4><p>{esc(text)}</p></div></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer"><div><b>Data Guardians</b> &nbsp;·&nbsp; Florida University Southeast capstone</div>'
    '<div>Customer Segmentation &amp; Personalization Engine <span class="g">◆</span> RFM + K-Means</div></div>',
    unsafe_allow_html=True,
)
