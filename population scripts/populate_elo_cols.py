from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

# EXCEL_PATH = r"data\group_stage_qualification_data.xlsx"
# SHEET_NAME = "Team_Group_Data"
data_folder_path = Path(__file__).parent.parent / "data"
EXCEL_PATH = data_folder_path / "Prediction Dataset_World Cup 2026.xlsx"
SHEET_NAME = "Dataset"

def compute_group_elo_features():
    # Read sheet (header is on row 3; skip first 2 rows)
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, skiprows=0)

    # Keep only rows with required fields
    needed = ["tournament_year", "group_id", "team_name", "team_elo_pre"]
    df = df[df[needed].notna().all(axis=1)].copy()

    df["tournament_year"] = df["tournament_year"].astype(int)
    df["group_id"] = df["group_id"].astype(str).str.strip()
    df["team_name"] = df["team_name"].astype(str).str.strip()
    df["team_elo_pre"] = pd.to_numeric(df["team_elo_pre"], errors="coerce")

    # --- Group-level stats (across ALL teams in the group) ---
    group_stats = (
        df.groupby(["tournament_year", "group_id"])["team_elo_pre"]
        .agg(group_elo_mean="mean", elo_std="std", group_size="count")
        .reset_index()
    )

    df = df.merge(group_stats, on=["tournament_year", "group_id"], how="left")

    #drop cols if already exist and make new
    if "opp_elo_mean" in df.columns:
        df.drop(columns=["opp_elo_mean"], inplace=True)
    if "opp_elo_max" in df.columns:
        df.drop(columns=["opp_elo_max"], inplace=True)
    if "group_elo_std" in df.columns:
        df.drop(columns=["group_elo_std"], inplace=True)
    if "elo_gap_vs_opp_mean" in df.columns:
        df.drop(columns=["elo_gap_vs_opp_mean"], inplace=True)
    

    # If std becomes NaN (rare; e.g., group_size==1), set to 0
    df["group_elo_std"] = df["elo_std"].fillna(0.0)

    # --- Opponent mean: (sum_all - team) / (n-1) ---
    group_sum = df["group_elo_mean"] * df["group_size"]
    df["opp_elo_mean"] = (group_sum - df["team_elo_pre"]) / (df["group_size"] - 1)

    # --- Opponent max: top1/top2 trick ---
    top2 = (
        df.groupby(["tournament_year", "group_id"])["team_elo_pre"]
        .apply(lambda s: pd.Series(sorted(s, reverse=True)[:2]))
        .unstack()
        .reset_index()
        .rename(columns={0: "top1", 1: "top2"})
    )
    df = df.merge(top2, on=["tournament_year", "group_id"], how="left")

    df["opp_elo_max"] = df.apply(
        lambda r: r["top2"] if r["team_elo_pre"] == r["top1"] else r["top1"],
        axis=1
    )

    # --- Elo gap ---
    df["elo_gap_vs_opp_mean"] = df["team_elo_pre"] - df["opp_elo_mean"]

    # Prepare output for writing back
    df_out = df[[
        "tournament_year", "group_id", "team_name",
        "opp_elo_mean", "opp_elo_max", "group_elo_std", "elo_gap_vs_opp_mean"
    ]].copy()

    # Lookup by (year, group_id, team_name)
    lookup = {}
    for r in df_out.itertuples(index=False):
        key = (int(r.tournament_year), str(r.group_id), str(r.team_name))
        lookup[key] = (
            float(r.opp_elo_mean),
            float(r.opp_elo_max),
            float(r.group_elo_std),
            float(r.elo_gap_vs_opp_mean),
        )

    # Write back to Excel in-place
    wb = load_workbook(EXCEL_PATH)
    ws = wb[SHEET_NAME]

    header_row = 1
    header_to_col = {}
    for col in range(1, ws.max_column + 1):
        v = ws.cell(header_row, col).value
        if v:
            header_to_col[str(v).strip()] = col

    required_cols = ["opp_elo_mean", "opp_elo_max", "group_elo_std", "elo_gap_vs_opp_mean"]

    for c in required_cols:
        if c not in header_to_col:
            #just add it at the end if missing
            new_col = ws.max_column + 1
            ws.cell(header_row, new_col).value = c
            header_to_col[c] = new_col
            # raise ValueError(f"Missing column in Excel header: {c}")

    data_start_row = 2
    rows_written = 0

    for excel_row in range(data_start_row, ws.max_row + 1):
        year = ws.cell(excel_row, header_to_col["tournament_year"]).value
        group_id = ws.cell(excel_row, header_to_col["group_id"]).value
        team = ws.cell(excel_row, header_to_col["team_name"]).value

        if year is None or group_id is None or team is None:
            continue

        key = (int(year), str(group_id).strip(), str(team).strip())
        if key not in lookup:
            continue

        opp_mean, opp_max, gstd, gap = lookup[key]
        ws.cell(excel_row, header_to_col["opp_elo_mean"]).value = opp_mean
        ws.cell(excel_row, header_to_col["opp_elo_max"]).value = opp_max
        ws.cell(excel_row, header_to_col["group_elo_std"]).value = gstd
        ws.cell(excel_row, header_to_col["elo_gap_vs_opp_mean"]).value = gap
        rows_written += 1

    try:
        wb.save(EXCEL_PATH)
    except PermissionError:
        raise PermissionError(f"Permission denied saving '{EXCEL_PATH}'. Close the Excel file and try again.")

    print(f"Done. Wrote opponent/group Elo features for {rows_written} rows into: {EXCEL_PATH}")

if __name__ == "__main__":
    compute_group_elo_features()
