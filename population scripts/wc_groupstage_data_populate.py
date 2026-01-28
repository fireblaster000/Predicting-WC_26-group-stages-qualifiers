"""
Populate the first 6 columns of the existing Excel template IN-PLACE using group_standings.csv.

Inputs:
- group_standings.csv
- group_stage_qualification_data.xlsx

Output:
- Overwrites group_stage_qualification_data.xlsx
"""

import pandas as pd
from openpyxl import load_workbook


# Paths (assumed relative to script location)

standings_path = "data/group_standings.csv"
output_path = "data/group_stage_qualification_data.xlsx"

# Settings

WC_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]

# Official tournament start dates (opening day / first match day)
# (Used only for the listed WC_YEARS)
start_date_map = {
    1998: "1998-06-10",
    2002: "2002-05-31",
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}


# Load & filter standings

gs = pd.read_csv(standings_path)

# Extract year from tournament_id (e.g., "WC-1998")
gs["tournament_year"] = gs["tournament_id"].astype(str).str.extract(r"(\d{4})").astype("Int64")

# Keep only:
# - tournament_id starting with WC-
# - stage_name == "group stage"
# - tournament_year in WC_YEARS
gs = gs[
    (gs["tournament_id"].astype(str).str.startswith("WC-")) &
    (gs["stage_name"].astype(str).str.lower() == "group stage") &
    (gs["tournament_year"].isin(WC_YEARS))
].copy()

# Map tournament start date
gs["tournament_start_date"] = gs["tournament_year"].map(start_date_map)

# Check: ensure no missing start dates for our year list
missing_dates = gs[gs["tournament_start_date"].isna()]["tournament_year"].unique().tolist()
if missing_dates:
    raise ValueError(f"Missing start dates for years: {missing_dates}. Update start_date_map.")

# Build output for first 6 columns
out = pd.DataFrame({
    "tournament_name": gs["tournament_name"],             # keep as-is from file
    "tournament_year": gs["tournament_year"].astype(int),
    "group_id": gs["group_name"],
    "team_name": gs["team_name"],
    "qualified_from_group": gs["advanced"].astype(int),   # 1/0 already
    "tournament_start_date": gs["tournament_start_date"],
}).sort_values(["tournament_year", "group_id", "team_name"]).reset_index(drop=True)


# Write into excel file IN-PLACE

wb = load_workbook(output_path)
ws = wb["Team_Group_Data"]

DATA_START_ROW = 4   # headers are at row 3
MAX_CLEAR_ROWS = 3000

# Clear existing values in first 6 cols
for r in range(DATA_START_ROW, DATA_START_ROW + MAX_CLEAR_ROWS):
    for c in range(1, 7):
        ws.cell(r, c).value = None

# Write new values
for i, row in enumerate(out.itertuples(index=False), start=DATA_START_ROW):
    ws.cell(i, 1).value = row.tournament_name
    ws.cell(i, 2).value = int(row.tournament_year)
    ws.cell(i, 3).value = row.group_id
    ws.cell(i, 4).value = row.team_name
    ws.cell(i, 5).value = int(row.qualified_from_group)
    ws.cell(i, 6).value = row.tournament_start_date  # YYYY-MM-DD string

# Save back to the SAME template file
wb.save(output_path)

print(f"Wrote {len(out)} rows into template (in-place): {output_path}")
print("Years included:", sorted(out["tournament_year"].unique().tolist()))
