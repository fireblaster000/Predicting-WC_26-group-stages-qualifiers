from pathlib import Path
import re
from openpyxl import load_workbook

# EXCEL_PATH = r"data\group_stage_qualification_data.xlsx"
# SHEET_NAME = "Team_Group_Data"
# change these if you move your Excel file or if the sheet name is different. The script will read the tournament_year and team_name columns, and write host_flag column.
data_folder_path = Path(__file__).parent.parent / "data"
EXCEL_PATH = data_folder_path / "Prediction Dataset_World Cup 2026.xlsx"
SHEET_NAME = "Dataset"  

# Header row is row 3 in your template
HEADER_ROW = 1
DATA_START_ROW = 2
# DATA_START_ROW = 4

# World Cup hosts (men) for 1998–2022
# NOTE: 2002 was co-hosted.
WC_HOSTS = {
    1998: ["France"],
    2002: ["Japan", "South Korea"],
    2006: ["Germany"],
    2010: ["South Africa"],
    2014: ["Brazil"],
    2018: ["Russia"],
    2022: ["Qatar"],
    2026: ["Mexico", "United States", "Canada"],
}

# Normalize team names in your sheet to match the host strings above
# Add to this if your sheet uses different spellings
NAME_MAP = {
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea Rep.": "South Korea",
    "IR Iran": "Iran",  # not needed for hosts, but safe to keep pattern
}

def norm(s: str) -> str:
    s = str(s).strip()
    s = NAME_MAP.get(s, s)
    s = re.sub(r"\s+", " ", s)
    return s

def add_or_get_column(ws, header_name: str) -> int:
    """Return column index for header_name. If missing, add it at the end."""
    header_to_col = {}
    for col in range(1, ws.max_column + 1):
        v = ws.cell(HEADER_ROW, col).value
        if v is not None:
            header_to_col[str(v).strip()] = col

    if header_name in header_to_col:
        return header_to_col[header_name]

    # Add new column at end
    new_col = ws.max_column + 1
    ws.cell(HEADER_ROW, new_col).value = header_name
    return new_col

def populate_host_flag():
    wb = load_workbook(EXCEL_PATH)
    ws = wb[SHEET_NAME]

    # Build header map
    header_to_col = {}
    for col in range(1, ws.max_column + 1):
        v = ws.cell(HEADER_ROW, col).value
        if v is not None:
            header_to_col[str(v).strip()] = col

    for needed in ["tournament_year", "team_name"]:
        if needed not in header_to_col:
            raise ValueError(f"Missing required column in header row: {needed}")

    year_col = header_to_col["tournament_year"]
    team_col = header_to_col["team_name"]

    host_flag_col = add_or_get_column(ws, "host_flag")

    rows_written = 0
    rows_seen = 0

    for r in range(DATA_START_ROW, ws.max_row + 1):
        year = ws.cell(r, year_col).value
        team = ws.cell(r, team_col).value

        if year is None or team is None:
            continue

        try:
            year = int(year)
        except Exception:
            continue

        team_norm = norm(team)
        hosts = [norm(x) for x in WC_HOSTS.get(year, [])]

        flag = 1 if team_norm in hosts else 0
        ws.cell(r, host_flag_col).value = flag

        rows_written += 1
        rows_seen += 1

    try:
        wb.save(EXCEL_PATH)
    except PermissionError:
        raise PermissionError(f"Permission denied saving '{EXCEL_PATH}'. Close the Excel file and try again.")

    print(f"Done. Wrote host_flag for {rows_written} rows into: {EXCEL_PATH}")

if __name__ == "__main__":
    populate_host_flag()
