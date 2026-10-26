"""
engine.py - analysis pipeline for the Data Guardians capstone dashboard.

Customer Segmentation and Personalization Engine for an Online Retail Store
Florida University Southeast (FUSE) - Business Intelligence and Data Analytics track

The pipeline follows the final report:
  1. load the Global Superstore data and standardize column names
  2. rebuild the corrupted Order Date from Year + weeknum (Monday of each week)
  3. aggregate to one row per Customer ID: Recency, Frequency (order records), Monetary (Sales)
  4. score each measure 1-5 by quintile (Recency reverse-scored); RFM total = 3-15
  5. standardize the three scores and run K-Means (K = 4, random_state = 42, n_init = 10)
  6. name the clusters from their mean RFM profile
  7. build the 4 x 3 cluster-by-account-type personalization matrix (12 strategies)
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Constants taken from the final report
# --------------------------------------------------------------------------------------
K_CLUSTERS = 4
RANDOM_STATE = 42
N_INIT = 10
STABILITY_SEEDS = (1, 7, 21, 99, 123)

CLUSTER_ORDER = ["Champions", "Loyal/Potential", "At-Risk", "Inactive"]
SEGMENT_ORDER = ["Consumer", "Corporate", "Home Office"]

CLUSTER_META = {
    "Champions": {
        "color": "#1F5D3A",
        "goal": "Reward and retain",
        "priority": 2,
        "focus": "Protect with loyalty reward tiers, volume rebates and priority support",
        "measures": "Retention rate; share of revenue retained",
    },
    "Loyal/Potential": {
        "color": "#6FB38B",
        "goal": "Grow",
        "priority": 3,
        "focus": "Grow through cross-sell and upsell offers, volume pricing and bundles",
        "measures": "Order frequency; average spend per customer",
    },
    "At-Risk": {
        "color": "#C9A227",
        "goal": "Re-engage first",
        "priority": 1,
        "focus": "Re-engage first: time-limited win-back offers with free shipping for Consumer, "
        "account-manager check-ins for Corporate, equipment upgrade credits for Home Office",
        "measures": "Reactivation rate; days since last order",
    },
    "Inactive": {
        "color": "#7A8795",
        "goal": "Win back at low cost",
        "priority": 4,
        "focus": "Win back at low cost with clearance discounts, low-cost trial orders and "
        "getting-started guides",
        "measures": "Reactivation rate; cost per reactivated customer",
    },
}

STRATEGIES = {
    ("Champions", "Consumer"): "Early access to flagship inventory and loyalty reward tiers",
    ("Champions", "Corporate"): "Volume-based corporate rebates and automated re-orders",
    ("Champions", "Home Office"): "Priority support and bundle pricing on business equipment",
    ("Loyal/Potential", "Consumer"): "Cross-sell and upsell offers based on purchase history",
    ("Loyal/Potential", "Corporate"): "Volume pricing for the next order tier",
    ("Loyal/Potential", "Home Office"): "Bundle offers for home-office setup",
    ("At-Risk", "Consumer"): "Time-limited win-back offer with free shipping",
    ("At-Risk", "Corporate"): "Account-manager check-in with a re-order incentive",
    ("At-Risk", "Home Office"): "Targeted win-back campaign with equipment upgrade credits",
    ("Inactive", "Consumer"): "High-margin clearance discounts to reactivate dormant accounts",
    ("Inactive", "Corporate"): "Re-engagement outreach with a low-cost trial order offer",
    ("Inactive", "Home Office"): "Getting-started guide with a comeback discount",
}

# Values reported in the final report (used by the "Report check" panel)
REPORT_REFERENCE = {
    "records": 51290,
    "accounts": 4873,
    "cluster_sizes": {"Champions": 1179, "Loyal/Potential": 1025, "At-Risk": 1255, "Inactive": 1414},
    "silhouette": 0.38,
    "chi2": 2.69,
    "p_value": 0.847,
    "cramers_v": 0.02,
    "revenue_share": {"Champions": 43.6, "Loyal/Potential": 10.0, "At-Risk": 38.9, "Inactive": 7.6},
    "recency_mean": 207.9,
    "recency_sd": 255.2,
}

REQUIRED_ALIASES = {
    "Customer ID": ["customer id", "customer_id", "customerid", "cust id"],
    "Customer Name": ["customer name", "customer_name", "customername"],
    "Segment": ["segment", "customer segment", "customer_segment"],
    "Sales": ["sales", "sales amount"],
    "Year": ["year"],
    "weeknum": ["weeknum", "week num", "week_num", "week number", "week"],
    "Order Date": ["order date", "order_date", "orderdate"],
    "Sub-Category": ["sub-category", "sub category", "sub_category", "subcategory"],
    "Category": ["category"],
    "Market": ["market"],
    "Region": ["region"],
    "Country": ["country"],
    "Profit": ["profit"],
    "Quantity": ["quantity"],
    "Order ID": ["order id", "order_id", "orderid"],
}


class DataError(ValueError):
    """Raised when the uploaded file cannot be used."""


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def read_table(raw: bytes, filename: str) -> pd.DataFrame:
    """Read a CSV or Excel file from bytes, trying common encodings for the CSV."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
        except UnicodeDecodeError as exc:  # try the next encoding
            last_error = exc
    raise DataError(f"Could not decode the file: {last_error}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical names used by the pipeline (case/space tolerant)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {c.lower(): c for c in df.columns}
    rename = {}
    for canonical, aliases in REQUIRED_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in [canonical.lower(), *aliases]:
            if alias in lookup and lookup[alias] not in rename:
                rename[lookup[alias]] = canonical
                break
    df = df.rename(columns=rename)
    missing = [c for c in ("Customer ID", "Segment", "Sales") if c not in df.columns]
    if missing:
        raise DataError(
            "Missing required column(s): " + ", ".join(missing)
            + ". The Global Superstore file needs Customer ID, Segment, Sales and either "
            "Year + weeknum or Order Date."
        )
    if not ({"Year", "weeknum"} <= set(df.columns) or "Order Date" in df.columns):
        raise DataError("Need either Year + weeknum, or an Order Date column.")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Customer ID", "Sales"])
    df["Customer ID"] = df["Customer ID"].astype(str).str.strip()
    df["Segment"] = df["Segment"].astype(str).str.strip()
    if "Customer Name" not in df.columns:
        df["Customer Name"] = df["Customer ID"]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------
def rebuild_order_dates(df: pd.DataFrame) -> pd.Series:
    """Monday of each recorded week, from Year + weeknum (report method, section 4.3.2).

    Week 53 of 2014 therefore falls on Monday 5 January 2015, which is why the latest
    rebuilt date is 5 January 2015 and the reference date is 6 January 2015.
    """
    pairs = df[["Year", "weeknum"]].drop_duplicates().copy()
    pairs["Year"] = pd.to_numeric(pairs["Year"], errors="coerce")
    pairs["weeknum"] = pd.to_numeric(pairs["weeknum"], errors="coerce")
    pairs = pairs.dropna()
    stamp = (
        pairs["Year"].astype(int).astype(str) + "-" + pairs["weeknum"].astype(int).astype(str) + "-1"
    )
    pairs["_date"] = pd.to_datetime(stamp, format="%Y-%W-%w", errors="coerce")
    merged = df[["Year", "weeknum"]].merge(pairs, on=["Year", "weeknum"], how="left")
    return pd.Series(merged["_date"].values, index=df.index)


def date_diagnostics(df: pd.DataFrame) -> dict:
    """Facts about the original Order Date column (shown in the data-quality panel)."""
    info: dict = {"has_order_date": "Order Date" in df.columns}
    if not info["has_order_date"]:
        return info
    parsed = pd.to_datetime(df["Order Date"], errors="coerce", format="mixed")
    info["distinct_original"] = int(parsed.nunique(dropna=True))
    info["original_min"] = parsed.min()
    info["original_max"] = parsed.max()
    if parsed.notna().any():
        last = parsed.groupby(df["Customer ID"]).max()
        ref = parsed.max() + pd.Timedelta(days=1)
        rec = (ref - last).dt.days
        info["recency_mean_original"] = float(rec.mean())
        info["recency_sd_original"] = float(rec.std(ddof=1)) if len(rec) > 1 else 0.0
    return info


# --------------------------------------------------------------------------------------
# RFM, clustering, naming
# --------------------------------------------------------------------------------------
def quintile_scores(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Quintile score 1-5. Ranks break ties, so every score gets one fifth of the customers."""
    ranks = values.rank(method="first", ascending=higher_is_better)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def build_rfm(tx: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    grouped = tx.groupby("Customer ID")
    rfm = pd.DataFrame(
        {
            "Customer Name": grouped["Customer Name"].first(),
            "Segment": grouped["Segment"].agg(lambda s: s.mode().iat[0]),
            "Last Order": grouped["OrderDate"].max(),
            "Frequency": grouped.size(),
            "Monetary": grouped["Sales"].sum(),
        }
    )
    rfm["Recency"] = (reference_date - rfm["Last Order"]).dt.days.astype(int)
    rfm["R"] = quintile_scores(rfm["Recency"], higher_is_better=False)  # recent = 5
    rfm["F"] = quintile_scores(rfm["Frequency"], higher_is_better=True)
    rfm["M"] = quintile_scores(rfm["Monetary"], higher_is_better=True)
    rfm["RFM"] = rfm["R"] + rfm["F"] + rfm["M"]
    return rfm.reset_index()


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Map raw K-Means labels to names using the mean RFM profile.

    Highest mean RFM total = Champions; lowest = Inactive. Of the two middle clusters, the one
    that has been silent longer (higher mean Recency in days) is At-Risk, the other is
    Loyal/Potential.
    """
    order = profile["RFM"].sort_values(ascending=False).index.tolist()
    champions, inactive = order[0], order[-1]
    middle = order[1:-1]
    at_risk = max(middle, key=lambda k: profile.loc[k, "Recency"])
    loyal = next(k for k in middle if k != at_risk)
    return {champions: "Champions", loyal: "Loyal/Potential", at_risk: "At-Risk", inactive: "Inactive"}


@dataclass
class Result:
    customers: pd.DataFrame
    transactions: pd.DataFrame
    profile: pd.DataFrame
    scaled: np.ndarray
    reference_date: pd.Timestamp
    silhouette: float
    date_info: dict = field(default_factory=dict)
    date_source: str = "rebuilt"
    focus: pd.DataFrame | None = None


def run_pipeline(df: pd.DataFrame, date_source: str = "rebuilt") -> Result:
    """Run the whole analysis. `date_source` is 'rebuilt' (Year + weeknum) or 'original'."""
    df = df.copy()
    info = date_diagnostics(df)
    if date_source == "rebuilt" and {"Year", "weeknum"} <= set(df.columns):
        df["OrderDate"] = rebuild_order_dates(df)
    elif "Order Date" in df.columns:
        df["OrderDate"] = pd.to_datetime(df["Order Date"], errors="coerce", format="mixed")
        date_source = "original"
    else:
        raise DataError("No usable date information in the file.")
    df = df.dropna(subset=["OrderDate"])
    if df.empty:
        raise DataError("All dates failed to parse.")

    reference_date = df["OrderDate"].max() + pd.Timedelta(days=1)
    rfm = build_rfm(df, reference_date)

    scaled = StandardScaler().fit_transform(rfm[["R", "F", "M"]].astype(float))
    model = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT).fit(scaled)
    rfm["Label"] = model.labels_

    raw_profile = rfm.groupby("Label").agg(Recency=("Recency", "mean"), RFM=("RFM", "mean"))
    names = name_clusters(raw_profile)
    rfm["Cluster"] = pd.Categorical(rfm["Label"].map(names), categories=CLUSTER_ORDER, ordered=True)
    rfm["Action"] = [
        STRATEGIES.get((c, s), "Review manually") for c, s in zip(rfm["Cluster"], rfm["Segment"])
    ]
    total_sales = rfm["Monetary"].sum()
    profile = (
        rfm.groupby("Cluster", observed=True)
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
        .reindex(CLUSTER_ORDER)
    )
    profile["Pct customers"] = profile["Customers"] / len(rfm) * 100
    profile["Pct revenue"] = profile["Revenue"] / total_sales * 100

    tx = df.merge(rfm[["Customer ID", "Cluster"]], on="Customer ID", how="left")
    sil = float(silhouette_score(scaled, model.labels_))
    result = Result(
        customers=rfm,
        transactions=tx,
        profile=profile,
        scaled=scaled,
        reference_date=reference_date,
        silhouette=sil,
        date_info=info,
        date_source=date_source,
    )
    if "Sub-Category" in tx.columns:
        result.focus = focus_subcategories(tx.merge(rfm[["Customer ID", "Segment"]].rename(
            columns={"Segment": "Account"}), on="Customer ID", how="left"))
    return result


# --------------------------------------------------------------------------------------
# Personalization, statistics, diagnostics
# --------------------------------------------------------------------------------------
def focus_subcategories(tx: pd.DataFrame) -> pd.DataFrame:
    """Sales by (cluster, account type, sub-category) with share and rank inside each cell."""
    table = (
        tx.groupby(["Cluster", "Account", "Sub-Category"], observed=True)["Sales"].sum().reset_index()
    )
    cell = table.groupby(["Cluster", "Account"], observed=True)["Sales"]
    table["Share"] = table["Sales"] / cell.transform("sum") * 100
    table["Rank"] = cell.rank(ascending=False, method="first").astype(int)
    return table.sort_values(["Cluster", "Account", "Rank"]).reset_index(drop=True)


def chi_square(customers: pd.DataFrame) -> dict:
    """Chi-square test of cluster against account type, with Cramer's V."""
    table = pd.crosstab(customers["Cluster"], customers["Segment"]).reindex(CLUSTER_ORDER)
    table = table.reindex(columns=[c for c in SEGMENT_ORDER if c in table.columns])
    chi2, p, dof, _ = chi2_contingency(table.values)
    n = table.values.sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if min(table.shape) > 1 else 0.0
    return {"table": table, "chi2": float(chi2), "p": float(p), "dof": int(dof), "n": int(n), "v": v}


def model_diagnostics(scaled: np.ndarray, labels: np.ndarray) -> dict:
    """Elbow curve, silhouette by K, and stability across random seeds."""
    elbow, silhouettes = [], []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(scaled)
        elbow.append({"K": k, "Inertia": float(km.inertia_)})
        if k >= 2:
            silhouettes.append({"K": k, "Silhouette": float(silhouette_score(scaled, km.labels_))})
    stability = []
    for seed in STABILITY_SEEDS:
        km = KMeans(n_clusters=K_CLUSTERS, random_state=seed, n_init=N_INIT).fit(scaled)
        stability.append({"Seed": seed, "Adjusted Rand index": float(adjusted_rand_score(labels, km.labels_))})
    return {
        "elbow": pd.DataFrame(elbow),
        "silhouette": pd.DataFrame(silhouettes),
        "stability": pd.DataFrame(stability),
    }


# --------------------------------------------------------------------------------------
# Demo data (synthetic, same schema as the Global Superstore file)
# --------------------------------------------------------------------------------------
_SUBCATS = {
    "Furniture": ["Bookcases", "Chairs", "Furnishings", "Tables"],
    "Office Supplies": ["Appliances", "Art", "Binders", "Envelopes", "Fasteners", "Labels", "Paper", "Storage", "Supplies"],
    "Technology": ["Accessories", "Copiers", "Machines", "Phones"],
}


def make_demo_data(seed: int = 7) -> pd.DataFrame:
    """Synthetic stand-in with the Global Superstore columns, so the app runs without the file.

    NOT real data: customer behavior is generated to resemble the report's four profiles.
    """
    rng = np.random.default_rng(seed)
    sizes = {"Champions": 1179, "Loyal/Potential": 1025, "At-Risk": 1255, "Inactive": 1414}
    profile = {  # recency days, frequency (records), monetary $
        "Champions": (35.2, 16.9, 4670.0),
        "Loyal/Potential": (50.7, 7.2, 1232.0),
        "At-Risk": (221.7, 14.1, 3915.0),
        "Inactive": (453.6, 4.5, 679.0),
    }
    shape = {"Champions": 2.0, "Loyal/Potential": 2.0, "At-Risk": 6.0, "Inactive": 8.0}
    sub_names = [s for subs in _SUBCATS.values() for s in subs]
    cat_of = {s: c for c, subs in _SUBCATS.items() for s in subs}
    boost = {
        "Champions": {"Phones": 5, "Chairs": 3, "Copiers": 2, "Bookcases": 2},
        "Loyal/Potential": {"Phones": 4, "Chairs": 3, "Storage": 3, "Bookcases": 2},
        "At-Risk": {"Phones": 4, "Copiers": 4, "Chairs": 3},
        "Inactive": {"Storage": 5, "Bookcases": 4, "Phones": 2},
    }
    monday0, monday_last = pd.Timestamp("2011-01-03"), pd.Timestamp("2014-12-29")
    weeks_total = int((monday_last - monday0).days // 7)
    ref = monday_last + pd.Timedelta(days=8)
    markets = ["APAC", "EU", "US", "LATAM", "Africa", "EMEA", "Canada"]
    rows = []
    cid = 10000
    for cluster, n in sizes.items():
        rec_mean, freq_mean, mon_mean = profile[cluster]
        w = np.ones(len(sub_names))
        for name, b in boost[cluster].items():
            w[sub_names.index(name)] += b
        w = w / w.sum()
        for _ in range(n):
            cid += 1
            segment = rng.choice(SEGMENT_ORDER, p=[0.515, 0.299, 0.186])
            recency = float(np.clip(rng.gamma(shape[cluster], rec_mean / shape[cluster]), 1, 1400))
            last = ref - pd.Timedelta(days=recency)
            last_idx = int(np.clip((last - monday0).days // 7, 0, weeks_total))
            k = int(1 + rng.poisson(max(freq_mean - 1, 0.1)))
            weeks = np.concatenate(([last_idx], rng.integers(0, last_idx + 1, size=k - 1)))
            total = float(rng.lognormal(np.log(mon_mean) - 0.125, 0.5))
            sales = rng.lognormal(0, 0.8, size=k)
            sales = sales / sales.sum() * total
            subs = rng.choice(sub_names, size=k, p=w)
            market = rng.choice(markets)
            for i in range(k):
                monday = monday0 + pd.Timedelta(weeks=int(weeks[i]))
                rows.append(
                    (
                        f"DM-{cid}",
                        f"Demo Customer {cid}",
                        segment,
                        float(sales[i]),
                        monday.year,
                        int(monday.strftime("%W")),
                        subs[i],
                        cat_of[subs[i]],
                        market,
                    )
                )
    out = pd.DataFrame(
        rows,
        columns=["Customer ID", "Customer Name", "Segment", "Sales", "Year", "weeknum", "Sub-Category", "Category", "Market"],
    )
    out["Order Date"] = "2011-01-01"  # mimics the corrupted field in the real file
    out["Order ID"] = [f"DM-O-{i}" for i in range(len(out))]
    out["Quantity"] = rng.integers(1, 8, size=len(out))
    out["Profit"] = out["Sales"] * rng.uniform(-0.1, 0.3, size=len(out))
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
