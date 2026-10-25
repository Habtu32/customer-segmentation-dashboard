# Customer Segmentation and Personalization Engine - Streamlit dashboard
Team Data Guardians, Florida University Southeast (FUSE) capstone.

## Run it (Windows, Mac or Linux)
1. Install Python 3.10 or newer.
2. Open a terminal in this folder and run:
       pip install -r requirements.txt
       streamlit run app.py
3. The browser opens at http://localhost:8501.

## Use the real data
- Upload the Global Superstore CSV/Excel file with the sidebar uploader,
- or copy the file next to app.py (its name must contain "superstore", e.g. Global Superstore.csv).
Without a file the app shows clearly labeled SYNTHETIC demo data.

Required columns: Customer ID, Segment, Sales, plus Year + weeknum (or Order Date).
Optional: Customer Name, Sub-Category (needed for the product focus in the strategy matrix).

## Method (same as the report)
Order Date is corrupted in this dataset, so dates are rebuilt as the Monday of each week from Year + weeknum;
reference date = latest rebuilt date + 1 day. RFM quintile scores 1-5 -> StandardScaler -> K-Means
(K=4, random_state=42, n_init=10). Clusters are named by rule from mean RFM and Recency.

## Check against the report
Open "Model & data quality". The "Check against the final report" table compares the app with the report
(4,873 accounts; clusters 1,179 / 1,025 / 1,255 / 1,414; silhouette 0.38; chi-square p = .847).
Small differences can come from how ties are broken when assigning quintiles.

## Files
app.py (interface), engine.py (data preparation, RFM, K-Means, statistics), .streamlit/config.toml (theme), assets/fuse_logo.png
