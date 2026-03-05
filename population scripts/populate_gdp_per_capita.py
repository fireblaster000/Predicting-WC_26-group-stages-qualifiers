import os
from pathlib import Path
import re
import math
import pandas as pd
from openpyxl import load_workbook

# =========================
# PATHS
# =========================
# EXCEL_PATH = r"data\group_stage_qualification_data.xlsx"
# SHEET_NAME = "Team_Group_Data"
data_folder_path = Path(__file__).parent.parent / "data"
EXCEL_PATH = data_folder_path / "Prediction Dataset_World Cup 2026.xlsx"
SHEET_NAME = "Dataset"

# Put your downloaded World Bank CSV here (you can move it into data/)
# WB_GDPPC_CSV = r"data\API_NY.GDP.PCAP.CD_DS2_en_csv_v2_31.csv"
WB_GDPPC_CSV = data_folder_path / "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_31.csv"

HEADER_ROW = 1
DATA_START_ROW = 2
# DATA_START_ROW = 4

YEAR_OFFSET = 1  # use tournament_year - 1
#if year 2026, we want GDP per capita from 2025 or latest available year before that. World Bank data is typically available up to 1-2 years before current year, so 2025 data may not be available yet. Adjust this offset as needed based on the latest year in your World Bank CSV.


# =========================
# TEAM -> WORLD BANK COUNTRY NAME MAPPING
# Expand if missing teams in the report.
# =========================
TEAM_TO_WB_NAME = {
    "Korea Republic": "Korea, Rep.",
    "South Korea": "Korea, Rep.",
    "IR Iran": "Iran, Islamic Rep.",
    "Iran": "Iran, Islamic Rep.",
    "Russia": "Russian Federation",
    "United States": "United States",
    "USA": "United States",
    "England": "United Kingdom",  
    "Wales": "United Kingdom",
    "Scotland": "United Kingdom",
    "Northern Ireland": "United Kingdom",
    "Ivory Coast": "Cote d'Ivoire",
    "Côte d’Ivoire": "Cote d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo, Dem. Rep.",
    "Congo DR": "Congo, Dem. Rep.",
    "Democratic Republic of the Congo": "Congo, Dem. Rep.",
    "Bolivia": "Bolivia",
    "Venezuela": "Venezuela, RB",
    "Syria": "Syrian Arab Republic",
    "Vietnam": "Viet Nam",
    "Laos": "Lao PDR",
    "Brunei": "Brunei Darussalam",
    "Czech Republic": "Czechia",
    "North Macedonia": "North Macedonia",
    "Slovakia": "Slovak Republic",  
    "Turkey": "Turkiye",
    "Serbia and Montenegro": "Serbia",
    "Egypt": "Egypt, Arab Rep.",
    "North Korea": "Korea, Dem. People's Rep.",
    "Republic of Ireland": "Ireland",
    "Yugoslavia": "Serbia",
}

def norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def team_to_wb_country(team_name: str) -> str:
    t = norm(team_name)
    return TEAM_TO_WB_NAME.get(t, t)

# =========================
# LOAD WORLD BANK CSV
# =========================
def load_worldbank_gdppc(csv_path: str) -> pd.DataFrame:
    """
    World Bank indicator download CSVs typically have 4 metadata rows at the top.
    We try skiprows=4; if it fails, fallback to no skip.
    Expected columns include:
      Country Name, Country Code, Indicator Name, Indicator Code, 1960, 1961, ...
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"World Bank CSV not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path, skiprows=4)
    except Exception:
        df = pd.read_csv(csv_path)

    # Basic sanity check
    required_cols = {"Country Name", "Country Code"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            "Could not parse World Bank CSV. "
            f"Columns found: {list(df.columns)[:20]}"
        )

    # Keep only real country rows (drop blanks)
    df = df.dropna(subset=["Country Name"]).copy()

    # Convert year columns to numeric where possible
    # (Some files include extra columns like 'Unnamed: ...' at end)
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build a long-form table: (country_name_norm, year) -> gdppc
    out = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdppc",
    )
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["country_name_norm"] = out["Country Name"].apply(lambda x: norm(x).lower())

    # Drop rows with no gdppc value
    out = out.dropna(subset=["year"]).copy()
    return out[["country_name_norm", "Country Code", "year", "gdppc"]]

def get_header_map(ws):
    m = {}
    for col in range(1, ws.max_column + 1):
        v = ws.cell(HEADER_ROW, col).value
        if v is not None:
            m[str(v).strip()] = col
    return m

def add_or_get_col(ws, header_to_col, name: str) -> int:
    if name in header_to_col:
        return header_to_col[name]
    new_col = ws.max_column + 1
    ws.cell(HEADER_ROW, new_col).value = name
    header_to_col[name] = new_col
    return new_col

# =========================
# MAIN: Populate GDP per capita
# =========================
def populate_gdppc_from_csv():
    # Load World Bank GDP per capita table
    wb_long = load_worldbank_gdppc(WB_GDPPC_CSV)

    # Fast lookup: (country_norm, year) -> gdppc
    gdp_lookup = {}
    for r in wb_long.itertuples(index=False):
        # r: country_name_norm, Country Code, year, gdppc
        if pd.notna(r.gdppc):
            gdp_lookup[(r.country_name_norm, int(r.year))] = float(r.gdppc)

    wb = load_workbook(EXCEL_PATH)
    ws = wb[SHEET_NAME]
    header_to_col = get_header_map(ws)

    for needed in ["tournament_year", "team_name"]:
        if needed not in header_to_col:
            raise ValueError(f"Missing required column in Excel header row: {needed}")

    year_col = header_to_col["tournament_year"]
    team_col = header_to_col["team_name"]

    # Output columns (create if missing)
    col_gdppc = add_or_get_col(ws, header_to_col, "gdp_per_capita_pre")
    col_log = add_or_get_col(ws, header_to_col, "log_gdp_per_capita_pre")

    missing_country_match = set()
    missing_value = []  # (tournament_year, team_name, macro_year, wb_country_used)

    rows_written = 0
    rows_seen = 0

    for row in range(DATA_START_ROW, ws.max_row + 1):
        tyear = ws.cell(row, year_col).value
        team = ws.cell(row, team_col).value

        if tyear is None or team is None:
            continue

        try:
            tyear = int(tyear)
        except Exception:
            continue
        #if year is 2026, we want GDP per capita from 2025 or latest available year before that. World Bank data is typically available up to 1-2 years before current year, so 2025 data may not be available yet. Adjust this offset as needed based on the latest year in your World Bank CSV.
        if tyear == 2026:
            macro_year = 2024  # or could set to 2024 if 2025 data is not available yet 
        else:
            macro_year = tyear - YEAR_OFFSET
        team = str(team).strip()

        wb_country = team_to_wb_country(team)
        wb_country_norm = norm(wb_country).lower()

        val = gdp_lookup.get((wb_country_norm, macro_year))

        rows_seen += 1

        if val is None or (isinstance(val, float) and math.isnan(val)):
            # maybe mapping mismatch, record it
            missing_value.append((tyear, team, macro_year, wb_country))
            # write blanks
            ws.cell(row, col_gdppc).value = None
            ws.cell(row, col_log).value = None
            continue

        # Write values
        ws.cell(row, col_gdppc).value = float(val)
        ws.cell(row, col_log).value = float(math.log(val)) if val > 0 else None
        rows_written += 1

    # Try save
    try:
        wb.save(EXCEL_PATH)
    except PermissionError:
        raise PermissionError(f"Permission denied saving '{EXCEL_PATH}'. Close the Excel file and rerun.")

    print(f"Done. Wrote GDP per capita (year-{YEAR_OFFSET}) for {rows_written}/{rows_seen} rows into {EXCEL_PATH}")

    # =========================
    # Missing report
    # =========================
    if missing_value:
        # group by team for readability
        miss_df = pd.DataFrame(missing_value, columns=["tournament_year", "team_name", "macro_year", "wb_country_used"])
        miss_df = miss_df.drop_duplicates().sort_values(["tournament_year", "team_name"])

        print("\nTeams/rows where GDP could NOT be written (either name mismatch or WB missing value for that year):")
        print(miss_df.to_string(index=False))

        print("\nNext step: if you see obvious naming issues, add them to TEAM_TO_WB_NAME at the top.")
    else:
        print("\nNo missing GDP rows. All teams matched and values found.")

if __name__ == "__main__":
    populate_gdppc_from_csv()
