import os
import re
import sys
import time
from datetime import datetime, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup
from openpyxl import load_workbook
from io import StringIO


# Paths (Edit these as needed)

EXCEL_PATH = r"data\group_stage_qualification_data.xlsx"  
SHEET_NAME = "Team_Group_Data"
CACHE_DIR = r"data\elo_snapshots_ifootball"  # snapshots saved here

os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# Elo snapshot URL template (site we are scraping from)
# Example:
# https://www.international-football.net/elo-ratings-table?year=1998&month=06&day=05
# -----------------------------
ELO_URL = "https://www.international-football.net/elo-ratings-table?year={year}&month={month:02d}&day={day:02d}"

# -----------------------------
# Optional: normalize names to improve matching
# We will likely need to expand this mapping as we see mismatches.
# Key = name in our Excel team_name; Value = name on international-football.net
# -----------------------------
NAME_MAP = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "United States": "United States", 
    "Republic of Ireland": "Ireland",
    # add more as needed
}

def normalize_team_name(s: str) -> str:
    s = str(s).strip()
    s = NAME_MAP.get(s, s)
    # common cleanup
    s = re.sub(r"\s+", " ", s)
    return s

def parse_elo_from_ifootball_html(html: str) -> pd.DataFrame:
    """
    Parse Elo snapshot from international-football.net elo-ratings-table pages.
    Extracts team name and elo rating from <tr class="survol"> rows.
    """
    soup = BeautifulSoup(html, "html.parser")

    rows = soup.select("tr.survol")
    pairs = []

    for row in rows:
        tds = row.find_all("td")
        # Expected layout:
        # tds[0] = flag cell (image)
        # tds[1] = team name (sometimes wrapped in <font>)
        # tds[2] = rating (numeric)
        if len(tds) < 3:
            continue

        # Team name cell may contain <font> or plain text
        team = tds[1].get_text(" ", strip=True)
        rating_txt = tds[2].get_text(" ", strip=True)

        # Some safety: rating must be 3-4 digits
        if not re.fullmatch(r"\d{3,4}", rating_txt):
            continue

        pairs.append((team, int(rating_txt)))

    if not pairs:
        raise ValueError("No Elo pairs parsed. Page structure may have changed or selector needs adjustment.")

    df = pd.DataFrame(pairs, columns=["team_name", "elo"])
    df = df.drop_duplicates(subset=["team_name"], keep="first").reset_index(drop=True)
    return df


def scrape_elo_snapshot(date_obj, sleep_s=0.5):
    date_str = date_obj.strftime("%Y-%m-%d")
    y, m, d = date_obj.year, date_obj.month, date_obj.day

    cache_path = os.path.join(CACHE_DIR, f"elo_snapshot_{date_str}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    url = ELO_URL.format(year=y, month=m, day=d)
    print(f"Fetching Elo snapshot for {date_str} from {url}...")

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    print("Status code:", r.status_code)
    r.raise_for_status()

    # parser is:
    snap = parse_elo_from_ifootball_html(r.text)

    snap["snapshot_date"] = date_str
    snap.to_csv(cache_path, index=False)

    time.sleep(sleep_s)
    return snap

def populate_team_elo_pre():
    # Read the sheet into pandas (template header is on row 3, so skip first 2 rows)
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=2)

    # Keep only rows that have the first 6 columns filled
    df = df[df["tournament_year"].notna() & df["team_name"].notna() & df["tournament_start_date"].notna()].copy()

    # Ensure proper types
    df["tournament_year"] = df["tournament_year"].astype(int)
    df["tournament_start_date"] = pd.to_datetime(df["tournament_start_date"], errors="coerce")

    # IMPORTANT: avoid duplicate columns from Excel
    # (If Excel already has 'team_elo_pre' column, drop it before we create a fresh one)
    if "team_elo_pre" in df.columns:
        df = df.drop(columns=["team_elo_pre"])

    # Compute cutoff_date_str = start_date - 1 day (NO to_pydatetime => no warning)
    df["cutoff_date_str"] = (df["tournament_start_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")

    # Normalize team names
    df["team_name_norm"] = df["team_name"].apply(normalize_team_name)

    # Scrape snapshots for all unique cutoff dates
    snapshots = []
    unique_dates = sorted(df["cutoff_date_str"].dropna().unique().tolist())

    print(f"Found {len(unique_dates)} unique Elo snapshot dates to fetch.")
    for date_str in unique_dates:
        # Convert back to datetime only for the scraper (if your scraper expects datetime)
        d = pd.to_datetime(date_str).to_pydatetime()

        snap = scrape_elo_snapshot(d)
        snap["team_name_norm"] = snap["team_name"].apply(normalize_team_name)

        # Keep only what we need
        snapshots.append(snap[["team_name_norm", "elo", "snapshot_date"]])

    elo_all = pd.concat(snapshots, ignore_index=True)

    # Merge Elo into df
    merged = df.merge(
        elo_all,
        left_on=["team_name_norm", "cutoff_date_str"],
        right_on=["team_name_norm", "snapshot_date"],
        how="left"
    )

    # Create team_elo_pre cleanly (no duplicate column name confusion)
    merged["team_elo_pre"] = pd.to_numeric(merged["elo"], errors="coerce")

    missing = int(merged["team_elo_pre"].isna().sum())
    print(f"Done. Missing Elo for {missing} rows (usually naming mismatches).")

    # Now write back to Excel in-place using openpyxl
    wb = load_workbook(EXCEL_PATH)
    ws = wb[SHEET_NAME]

    # Find column index for 'team_elo_pre' in the file header row (row 3)
    header_row = 3
    header_to_col = {}
    for col in range(1, ws.max_column + 1):
        val = ws.cell(header_row, col).value
        if val:
            header_to_col[str(val).strip()] = col

    if "team_elo_pre" not in header_to_col:
        raise ValueError("Could not find 'team_elo_pre' column in the Excel template header row.")

    elo_col = header_to_col["team_elo_pre"]

    # Build a lookup from (year, group_id, team_name) -> elo
    merged["group_id"] = merged["group_id"].astype(str).str.strip()
    merged["team_name"] = merged["team_name"].astype(str).str.strip()

    elo_lookup = {}
    for r in merged.itertuples(index=False):
        key = (int(getattr(r, "tournament_year")), str(getattr(r, "group_id")), str(getattr(r, "team_name")))
        elo_lookup[key] = getattr(r, "team_elo_pre")

    # Write Elo values row-by-row (data starts at row 4)
    data_start_row = 4
    rows_written = 0

    for excel_row in range(data_start_row, ws.max_row + 1):
        year = ws.cell(excel_row, header_to_col["tournament_year"]).value
        group_id = ws.cell(excel_row, header_to_col["group_id"]).value
        team = ws.cell(excel_row, header_to_col["team_name"]).value

        if year is None or team is None or group_id is None:
            continue

        key = (int(year), str(group_id).strip(), str(team).strip())
        if key in elo_lookup:
            val = elo_lookup[key]
            ws.cell(excel_row, elo_col).value = float(val) if pd.notna(val) else None
            rows_written += 1

    wb.save(EXCEL_PATH)
    print(f"Wrote team_elo_pre into {rows_written} Excel rows (in-place). File: {EXCEL_PATH}")

    # Optional: print which teams are missing (debug)
    if missing > 0:
        miss_df = merged.loc[merged["team_elo_pre"].isna(), ["tournament_year", "team_name"]].drop_duplicates()
        print("Teams missing Elo (check naming):")
        print(miss_df.to_string(index=False))

if __name__ == "__main__":
    populate_team_elo_pre()
