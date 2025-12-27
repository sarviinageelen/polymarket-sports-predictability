#!/usr/bin/env python3
"""
Generate actionable betting insights from Polymarket data.

Creates an Excel workbook with 7 focused tabs:
1. Index: Overview of all sheets + key takeaways
2. Quick Reference: Top actionable strategies at a glance
3. Sport Guide: Which sports to bet + recommendations
4. Underdog Opportunities: Best underdog bets by sport/threshold
5. Reliable Favorites: Teams/players with best win rates as favorites
6. Market Efficiency: Calibration data proving market accuracy
7. Raw Data: All sports summary stats for reference

Output:
- Excel workbook: outputs/favourite_win_rates.xlsx
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import xlsxwriter

# Use absolute paths to work from any directory
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "fetch_events.csv"
EXCEL_OUTPUT_FILE = PROJECT_ROOT / "outputs" / "favourite_win_rates.xlsx"

# Sport display names (code -> display name)
SPORT_DISPLAY_NAMES = {
    "atp": "ATP",
    "wta": "WTA",
    "nba": "NBA",
    "nfl": "NFL",
    "mlb": "MLB",
    "cfb": "CFB",
    "ncaab": "CBB",
    "cbb": "CBB",
}

# Excel styling
HEADER_COLOR = "#3B8686"  # Teal color for headers


def load_data(filename):
    """Load event data from CSV file with schema validation."""
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(
            f"CSV file not found: {filename}\n"
            f"Please run fetch_events.py first to generate the data."
        )

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} events from {filename}")

    required_columns = [
        "sport", "closed", "p1_close", "p2_close",
        "outcome_1", "outcome_2", "winner"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"CSV schema validation failed. Missing required columns: {missing_columns}"
        )

    return df


def filter_by_year(df, year):
    """Filter dataframe to events from a specific year based on game_start_time."""
    if "game_start_time" not in df.columns:
        print(f"Warning: game_start_time column not found, returning all data")
        return df

    def get_year(date_str):
        if pd.isna(date_str) or not str(date_str).strip():
            return None
        try:
            dt = datetime.fromisoformat(str(date_str).replace(" ", "T").split(".")[0])
            return dt.year
        except (ValueError, AttributeError):
            return None

    df_copy = df.copy()
    df_copy["_year"] = df_copy["game_start_time"].apply(get_year)
    filtered = df_copy[df_copy["_year"] == year].drop(columns=["_year"])
    print(f"Filtered to {len(filtered)} events from {year}")
    return filtered


def calculate_favourite_win_rates(df, verbose=True):
    """Calculate favourite win rate for each sport.

    Returns a dict: {sport_code: (win_rate, wins, total)}
    """
    closed_df = df[df["closed"] == 1].copy()
    if verbose:
        print(f"Analyzing {len(closed_df)} closed events")

    closed_df["sport"] = closed_df["sport"].replace({"ncaab": "cbb"})

    results = {}

    for sport in closed_df["sport"].unique():
        sport_df = closed_df[closed_df["sport"] == sport]

        if len(sport_df) == 0:
            continue

        favourite_wins = 0
        total = 0

        for _, row in sport_df.iterrows():
            p1_close = row["p1_close"]
            p2_close = row["p2_close"]
            winner = row["winner"]
            outcome_1 = row["outcome_1"]
            outcome_2 = row["outcome_2"]

            if pd.isna(p1_close) or pd.isna(p2_close) or pd.isna(winner):
                continue

            if not str(winner).strip():
                continue

            if p1_close > p2_close:
                favourite = outcome_1
            elif p2_close > p1_close:
                favourite = outcome_2
            else:
                continue

            if favourite.strip() == winner.strip():
                favourite_wins += 1

            total += 1

        if total > 0:
            win_rate = (favourite_wins / total) * 100
            results[sport] = (win_rate, favourite_wins, total)
            if verbose:
                print(f"  {sport.upper()}: {win_rate:.1f}% ({favourite_wins}/{total})")

    return results


def prepare_analysis_data(df):
    """Prepare closed events with favorite/underdog analysis fields."""
    closed = df[(df["closed"] == 1) &
                (df["p1_close"].notna()) &
                (df["p2_close"].notna())].copy()

    closed["sport"] = closed["sport"].replace({"ncaab": "cbb"})

    def get_analysis(row):
        p1, p2 = row["p1_close"], row["p2_close"]
        winner = row["winner"]
        o1, o2 = row["outcome_1"], row["outcome_2"]

        if p1 > p2:
            fav, fav_price, dog_price = o1, p1, p2
        else:
            fav, fav_price, dog_price = o2, p2, p1

        fav_won = str(fav).strip() == str(winner).strip() if pd.notna(winner) else False
        return pd.Series({
            "favorite": fav,
            "fav_price": fav_price,
            "dog_price": dog_price,
            "fav_won": fav_won,
            "upset": not fav_won
        })

    result = closed.apply(get_analysis, axis=1)
    for col in result.columns:
        closed[col] = result[col]

    return closed


def calculate_market_calibration(df):
    """Calculate win rates by price bucket to show market calibration."""
    closed = prepare_analysis_data(df)

    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%",
              "75-80%", "80-85%", "85-90%", "90-95%", "95-100%"]

    closed["bucket"] = pd.cut(closed["fav_price"], bins=bins, labels=labels)

    results = []
    for i, label in enumerate(labels):
        bucket_df = closed[closed["bucket"] == label]
        if len(bucket_df) == 0:
            continue

        expected = (bins[i] + bins[i + 1]) / 2
        actual = bucket_df["fav_won"].mean()
        calibration_error = actual - expected

        results.append({
            "bucket": label,
            "expected": expected * 100,
            "actual": actual * 100,
            "sample_size": len(bucket_df),
            "calibration_error": calibration_error * 100
        })

    return results


def calculate_underdog_value(df):
    """Calculate ROI of betting on underdogs by sport at different thresholds."""
    closed = prepare_analysis_data(df)

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]
    results = []

    for sport in sorted(closed["sport"].unique()):
        sport_df = closed[closed["sport"] == sport]

        for threshold in thresholds:
            dogs = sport_df[sport_df["dog_price"] >= threshold]
            if len(dogs) < 20:
                continue

            wins = len(dogs[dogs["upset"]])
            total_bet = len(dogs) * 100

            total_return = sum(
                100 / row["dog_price"]
                for _, row in dogs[dogs["upset"]].iterrows()
            )
            roi = ((total_return - total_bet) / total_bet) * 100

            # Recommendation based on ROI
            if roi > 3:
                recommendation = "BUY"
            elif roi > 0:
                recommendation = "CONSIDER"
            elif roi > -3:
                recommendation = "NEUTRAL"
            else:
                recommendation = "AVOID"

            results.append({
                "sport": SPORT_DISPLAY_NAMES.get(sport, sport.upper()),
                "threshold": f"{int(threshold * 100)}%+",
                "bets": len(dogs),
                "wins": wins,
                "win_rate": (wins / len(dogs)) * 100,
                "roi": roi,
                "recommendation": recommendation
            })

    return results


def calculate_sport_guide(df):
    """Calculate comprehensive sport guide with recommendations."""
    closed = prepare_analysis_data(df)

    results = []
    for sport in sorted(closed["sport"].unique()):
        sport_df = closed[closed["sport"] == sport]
        if len(sport_df) < 50:
            continue

        display_name = SPORT_DISPLAY_NAMES.get(sport, sport.upper())

        # Favorite win rate
        fav_win_rate = sport_df["fav_won"].mean() * 100

        # Underdog ROI (at 30% threshold)
        dogs = sport_df[sport_df["dog_price"] >= 0.30]
        if len(dogs) >= 20:
            total_bet = len(dogs) * 100
            total_return = sum(100 / r["dog_price"] for _, r in dogs[dogs["upset"]].iterrows())
            dog_roi = ((total_return - total_bet) / total_bet) * 100
            dog_sample = len(dogs)
        else:
            dog_roi = None
            dog_sample = 0

        # Recommendation logic
        if dog_roi is not None and dog_roi > 3:
            recommendation = "Bet Underdogs"
            action = "Target underdogs priced 30%+"
        elif fav_win_rate > 75:
            recommendation = "Favor Favorites"
            action = "Stick with heavy favorites"
        elif dog_roi is not None and dog_roi < -3:
            recommendation = "Avoid Underdogs"
            action = "Markets are efficient here"
        else:
            recommendation = "Neutral"
            action = "No clear edge"

        results.append({
            "sport": display_name,
            "fav_win_rate": fav_win_rate,
            "dog_roi": dog_roi,
            "dog_sample": dog_sample,
            "total_events": len(sport_df),
            "recommendation": recommendation,
            "action": action
        })

    # Sort by dog ROI descending
    results.sort(key=lambda x: x["dog_roi"] if x["dog_roi"] is not None else -999, reverse=True)
    return results


def calculate_reliable_favorites(df):
    """Calculate most reliable teams/players when favored per sport."""
    closed = prepare_analysis_data(df)

    results = {}
    for sport in sorted(closed["sport"].unique()):
        sport_df = closed[closed["sport"] == sport]

        team_stats = sport_df.groupby("favorite").agg({
            "fav_won": ["sum", "count", "mean"],
            "fav_price": "mean"
        })
        team_stats.columns = ["wins", "total", "win_rate", "avg_price"]
        team_stats = team_stats[team_stats["total"] >= 5]  # Min 5 games
        team_stats = team_stats.sort_values("win_rate", ascending=False).head(10)

        # Calculate ROI for each team
        display_name = SPORT_DISPLAY_NAMES.get(sport, sport.upper())
        team_list = []
        for team, row in team_stats.iterrows():
            # ROI = (total_return - total_bet) / total_bet
            total_bet = int(row["total"]) * 100
            wins = int(row["wins"])
            total_return = wins * (100 / row["avg_price"])
            roi = ((total_return - total_bet) / total_bet) * 100

            team_list.append({
                "team": team,
                "wins": wins,
                "total": int(row["total"]),
                "win_rate": row["win_rate"] * 100,
                "avg_price": row["avg_price"] * 100,
                "roi": roi
            })

        results[display_name] = team_list

    return results


def calculate_quick_reference(sport_guide, underdog_data):
    """Generate quick reference data from other analyses."""
    # Top 3 profitable strategies (from underdog data with positive ROI)
    profitable = sorted(
        [u for u in underdog_data if u["roi"] > 0],
        key=lambda x: x["roi"],
        reverse=True
    )[:3]

    # Top 3 strategies to avoid
    avoid = sorted(
        [u for u in underdog_data if u["roi"] < -2],
        key=lambda x: x["roi"]
    )[:3]

    # Best and worst sports
    best_sport = sport_guide[0] if sport_guide else None
    worst_sport = sport_guide[-1] if sport_guide else None

    return {
        "profitable": profitable,
        "avoid": avoid,
        "best_sport": best_sport,
        "worst_sport": worst_sport
    }


def create_excel_output(win_rates_2025, win_rates_all, calibration_data,
                        underdog_data, sport_guide_data, reliable_favorites_data,
                        quick_ref_data, output_file):
    """Create Excel workbook with 7 focused tabs."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(str(output_path))

    # Define formats
    title_format = workbook.add_format({
        'bold': True,
        'font_size': 18,
        'font_color': '#1a1a1a'
    })
    section_format = workbook.add_format({
        'bold': True,
        'font_size': 14,
        'font_color': '#333333',
        'bottom': 1
    })
    header_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'font_color': 'white',
        'bg_color': HEADER_COLOR,
        'align': 'center',
        'border': 1
    })
    cell_format = workbook.add_format({'border': 1})
    percent_format = workbook.add_format({'border': 1, 'num_format': '0.0'})
    roi_positive = workbook.add_format({'border': 1, 'num_format': '+0.00', 'font_color': '#006600'})
    roi_negative = workbook.add_format({'border': 1, 'num_format': '0.00', 'font_color': '#CC0000'})
    buy_format = workbook.add_format({'border': 1, 'bold': True, 'font_color': '#006600', 'bg_color': '#e6ffe6'})
    avoid_format = workbook.add_format({'border': 1, 'bold': True, 'font_color': '#CC0000', 'bg_color': '#ffe6e6'})
    neutral_format = workbook.add_format({'border': 1, 'font_color': '#666666'})
    sport_header_format = workbook.add_format({
        'bold': True,
        'font_size': 14,
        'bg_color': '#E8E8E8',
        'border': 1
    })
    note_format = workbook.add_format({'italic': True, 'font_color': '#666666', 'font_size': 10})
    key_insight_format = workbook.add_format({
        'bold': True,
        'font_size': 11,
        'font_color': '#1a5276',
        'text_wrap': True
    })

    # === Sheet 1: Index ===
    ws_index = workbook.add_worksheet("Index")
    ws_index.set_column(0, 0, 40)
    ws_index.set_column(1, 1, 60)

    row = 0
    ws_index.write(row, 0, "POLYMARKET SPORTS BETTING INSIGHTS", title_format)
    row += 2

    # Data summary
    total_events = sum(v[2] for v in win_rates_all.values())

    ws_index.write(row, 0, "DATA SUMMARY", section_format)
    row += 1
    ws_index.write(row, 0, f"Total events analyzed: {total_events:,}")
    row += 1
    ws_index.write(row, 0, f"Sports covered: {len(win_rates_all)}")
    row += 1
    ws_index.write(row, 0, f"Data period: 2024-2025")
    row += 2

    # Key takeaways
    ws_index.write(row, 0, "KEY TAKEAWAYS", section_format)
    row += 1

    takeaways = []
    if quick_ref_data["profitable"]:
        best = quick_ref_data["profitable"][0]
        takeaways.append(f"1. Best opportunity: {best['sport']} underdogs ({best['threshold']}) -> +{best['roi']:.1f}% ROI")
    if len(quick_ref_data["profitable"]) > 1:
        second = quick_ref_data["profitable"][1]
        takeaways.append(f"2. Second best: {second['sport']} underdogs ({second['threshold']}) -> +{second['roi']:.1f}% ROI")
    if quick_ref_data["avoid"]:
        worst = quick_ref_data["avoid"][0]
        takeaways.append(f"3. Avoid: {worst['sport']} underdogs -> {worst['roi']:.1f}% ROI")
    takeaways.append("4. Market is well-calibrated - no easy edge from naive strategies")

    for takeaway in takeaways:
        ws_index.write(row, 0, takeaway, key_insight_format)
        row += 1
    row += 1

    # Sheet guide
    ws_index.write(row, 0, "SHEET GUIDE", section_format)
    row += 1

    sheets = [
        ("Quick Reference", "Start here for actionable betting decisions"),
        ("Sport Guide", "Which sports to focus on with specific recommendations"),
        ("Underdog Opportunities", "Specific underdog betting thresholds by sport"),
        ("Reliable Favorites", "Teams that deliver when favored"),
        ("Market Efficiency", "Proof that markets are well-calibrated"),
        ("Raw Data", "Detailed statistics by sport for reference"),
    ]

    for sheet, desc in sheets:
        ws_index.write(row, 0, f"  {sheet}")
        ws_index.write(row, 1, desc)
        row += 1

    row += 2
    ws_index.write(row, 0, "Source: Polymarket closing prices", note_format)

    # === Sheet 2: Quick Reference ===
    ws_quick = workbook.add_worksheet("Quick Reference")
    ws_quick.set_column(0, 0, 25)
    ws_quick.set_column(1, 1, 45)
    ws_quick.set_column(2, 2, 15)
    ws_quick.set_column(3, 3, 15)

    row = 0
    ws_quick.write(row, 0, "ACTIONABLE BETTING INSIGHTS", title_format)
    row += 2

    # Top profitable strategies
    ws_quick.write(row, 0, "TOP PROFITABLE STRATEGIES", section_format)
    row += 1

    if quick_ref_data["profitable"]:
        headers = ["Strategy", "Description", "ROI (%)", "Sample"]
        for col, h in enumerate(headers):
            ws_quick.write(row, col, h, header_format)
        row += 1

        for i, p in enumerate(quick_ref_data["profitable"], 1):
            ws_quick.write(row, 0, f"{i}. {p['sport']} Underdogs", cell_format)
            ws_quick.write(row, 1, f"Bet underdogs priced {p['threshold']}", cell_format)
            ws_quick.write(row, 2, p["roi"], roi_positive)
            ws_quick.write(row, 3, p["bets"], cell_format)
            row += 1
    else:
        ws_quick.write(row, 0, "No profitable strategies found", note_format)
        row += 1

    row += 1

    # Strategies to avoid
    ws_quick.write(row, 0, "STRATEGIES TO AVOID", section_format)
    row += 1

    if quick_ref_data["avoid"]:
        headers = ["Strategy", "Description", "ROI (%)", "Sample"]
        for col, h in enumerate(headers):
            ws_quick.write(row, col, h, header_format)
        row += 1

        for i, a in enumerate(quick_ref_data["avoid"], 1):
            ws_quick.write(row, 0, f"{i}. {a['sport']} Underdogs", cell_format)
            ws_quick.write(row, 1, f"Avoid underdogs priced {a['threshold']}", cell_format)
            ws_quick.write(row, 2, a["roi"], roi_negative)
            ws_quick.write(row, 3, a["bets"], cell_format)
            row += 1
    else:
        ws_quick.write(row, 0, "No strategies to avoid found", note_format)
        row += 1

    row += 1

    # Best/worst sport summary
    ws_quick.write(row, 0, "SPORT SUMMARY", section_format)
    row += 1

    if quick_ref_data["best_sport"]:
        best = quick_ref_data["best_sport"]
        ws_quick.write(row, 0, "Best sport for underdogs:", cell_format)
        ws_quick.write(row, 1, f"{best['sport']} ({best['recommendation']})", buy_format)
        row += 1

    if quick_ref_data["worst_sport"]:
        worst = quick_ref_data["worst_sport"]
        ws_quick.write(row, 0, "Worst sport for underdogs:", cell_format)
        ws_quick.write(row, 1, f"{worst['sport']} ({worst['recommendation']})", avoid_format)
        row += 1

    row += 2
    ws_quick.write(row, 0, "OVERALL RECOMMENDATION", section_format)
    row += 1
    ws_quick.write(row, 0, "Markets are efficient. Focus on MLB/ATP underdogs for best edge.", key_insight_format)

    # === Sheet 3: Sport Guide ===
    ws_sport = workbook.add_worksheet("Sport Guide")
    ws_sport.set_column(0, 0, 10)
    ws_sport.set_column(1, 1, 16)
    ws_sport.set_column(2, 2, 16)
    ws_sport.set_column(3, 3, 14)
    ws_sport.set_column(4, 4, 12)
    ws_sport.set_column(5, 5, 18)
    ws_sport.set_column(6, 6, 30)

    headers = ["Sport", "Fav Win Rate (%)", "Underdog ROI (%)", "Dog Sample", "Events", "Recommendation", "Action"]
    for col, h in enumerate(headers):
        ws_sport.write(0, col, h, header_format)

    for row_idx, item in enumerate(sport_guide_data, 1):
        ws_sport.write(row_idx, 0, item["sport"], cell_format)
        ws_sport.write(row_idx, 1, round(item["fav_win_rate"], 1), percent_format)

        if item["dog_roi"] is not None:
            roi_fmt = roi_positive if item["dog_roi"] > 0 else roi_negative
            ws_sport.write(row_idx, 2, round(item["dog_roi"], 2), roi_fmt)
        else:
            ws_sport.write(row_idx, 2, "N/A", cell_format)

        ws_sport.write(row_idx, 3, item["dog_sample"], cell_format)
        ws_sport.write(row_idx, 4, item["total_events"], cell_format)

        rec = item["recommendation"]
        if rec == "Bet Underdogs":
            rec_fmt = buy_format
        elif rec == "Avoid Underdogs":
            rec_fmt = avoid_format
        else:
            rec_fmt = neutral_format
        ws_sport.write(row_idx, 5, rec, rec_fmt)
        ws_sport.write(row_idx, 6, item["action"], cell_format)

    if sport_guide_data:
        ws_sport.add_table(0, 0, len(sport_guide_data), 6, {
            'name': 'SportGuide',
            'columns': [{'header': h} for h in headers],
            'style': 'Table Style Medium 9'
        })

    # === Sheet 4: Underdog Opportunities ===
    ws_underdog = workbook.add_worksheet("Underdog Opportunities")
    ws_underdog.set_column(0, 0, 10)
    ws_underdog.set_column(1, 1, 12)
    ws_underdog.set_column(2, 2, 10)
    ws_underdog.set_column(3, 3, 10)
    ws_underdog.set_column(4, 4, 14)
    ws_underdog.set_column(5, 5, 12)
    ws_underdog.set_column(6, 6, 16)

    headers = ["Sport", "Threshold", "Bets", "Wins", "Win Rate (%)", "ROI (%)", "Recommendation"]
    for col, h in enumerate(headers):
        ws_underdog.write(0, col, h, header_format)

    for row_idx, item in enumerate(underdog_data, 1):
        ws_underdog.write(row_idx, 0, item["sport"], cell_format)
        ws_underdog.write(row_idx, 1, item["threshold"], cell_format)
        ws_underdog.write(row_idx, 2, item["bets"], cell_format)
        ws_underdog.write(row_idx, 3, item["wins"], cell_format)
        ws_underdog.write(row_idx, 4, round(item["win_rate"], 1), percent_format)

        roi_fmt = roi_positive if item["roi"] > 0 else roi_negative
        ws_underdog.write(row_idx, 5, round(item["roi"], 2), roi_fmt)

        rec = item["recommendation"]
        if rec == "BUY":
            rec_fmt = buy_format
        elif rec == "AVOID":
            rec_fmt = avoid_format
        else:
            rec_fmt = neutral_format
        ws_underdog.write(row_idx, 6, rec, rec_fmt)

    if underdog_data:
        ws_underdog.add_table(0, 0, len(underdog_data), 6, {
            'name': 'UnderdogOpportunities',
            'columns': [{'header': h} for h in headers],
            'style': 'Table Style Medium 9'
        })

    # === Sheet 5: Reliable Favorites ===
    ws_favorites = workbook.add_worksheet("Reliable Favorites")
    ws_favorites.set_column(0, 0, 30)
    ws_favorites.set_column(1, 1, 10)
    ws_favorites.set_column(2, 2, 10)
    ws_favorites.set_column(3, 3, 14)
    ws_favorites.set_column(4, 4, 14)
    ws_favorites.set_column(5, 5, 12)

    row = 0
    for sport, teams in reliable_favorites_data.items():
        if not teams:
            continue

        ws_favorites.merge_range(row, 0, row, 5, sport, sport_header_format)
        row += 1

        headers = ["Team/Player", "Wins", "Total", "Win Rate (%)", "Avg Price (%)", "ROI (%)"]
        for col, h in enumerate(headers):
            ws_favorites.write(row, col, h, header_format)
        row += 1

        for team in teams:
            ws_favorites.write(row, 0, team["team"], cell_format)
            ws_favorites.write(row, 1, team["wins"], cell_format)
            ws_favorites.write(row, 2, team["total"], cell_format)
            ws_favorites.write(row, 3, round(team["win_rate"], 1), percent_format)
            ws_favorites.write(row, 4, round(team["avg_price"], 0), percent_format)

            roi_fmt = roi_positive if team["roi"] > 0 else roi_negative
            ws_favorites.write(row, 5, round(team["roi"], 2), roi_fmt)
            row += 1

        row += 1

    # === Sheet 6: Market Efficiency ===
    ws_calibration = workbook.add_worksheet("Market Efficiency")
    ws_calibration.set_column(0, 0, 14)
    ws_calibration.set_column(1, 4, 16)

    headers = ["Price Bucket", "Expected (%)", "Actual (%)", "Sample Size", "Calibration Error"]
    for col, h in enumerate(headers):
        ws_calibration.write(0, col, h, header_format)

    for row_idx, item in enumerate(calibration_data, 1):
        ws_calibration.write(row_idx, 0, item["bucket"], cell_format)
        ws_calibration.write(row_idx, 1, round(item["expected"], 1), percent_format)
        ws_calibration.write(row_idx, 2, round(item["actual"], 1), percent_format)
        ws_calibration.write(row_idx, 3, item["sample_size"], cell_format)
        ws_calibration.write(row_idx, 4, round(item["calibration_error"], 2), percent_format)

    if calibration_data:
        ws_calibration.add_table(0, 0, len(calibration_data), 4, {
            'name': 'MarketEfficiency',
            'columns': [{'header': h} for h in headers],
            'style': 'Table Style Medium 9'
        })

    # Add interpretation note
    row = len(calibration_data) + 3
    ws_calibration.write(row, 0, "INTERPRETATION", section_format)
    row += 1
    ws_calibration.write(row, 0, "Calibration error close to 0 = efficient market", note_format)
    row += 1
    ws_calibration.write(row, 0, "Positive error = favorites win more than price suggests", note_format)
    row += 1
    ws_calibration.write(row, 0, "Negative error = favorites win less than price suggests", note_format)

    # === Sheet 7: Raw Data ===
    ws_raw = workbook.add_worksheet("Raw Data")
    ws_raw.set_column(0, 0, 12)
    ws_raw.set_column(1, 3, 16)

    row = 0
    ws_raw.write(row, 0, "2025 DATA", section_format)
    row += 1

    headers = ["Sport", "Favorite Wins", "Total Events", "Win Rate (%)"]
    for col, h in enumerate(headers):
        ws_raw.write(row, col, h, header_format)
    row += 1

    sorted_2025 = sorted(win_rates_2025.items(), key=lambda x: x[1][0], reverse=True)
    for sport_code, (win_rate, wins, total) in sorted_2025:
        display_name = SPORT_DISPLAY_NAMES.get(sport_code, sport_code.upper())
        ws_raw.write(row, 0, display_name, cell_format)
        ws_raw.write(row, 1, wins, cell_format)
        ws_raw.write(row, 2, total, cell_format)
        ws_raw.write(row, 3, round(win_rate, 1), percent_format)
        row += 1

    row += 2
    ws_raw.write(row, 0, "ALL YEARS DATA", section_format)
    row += 1

    for col, h in enumerate(headers):
        ws_raw.write(row, col, h, header_format)
    row += 1

    sorted_all = sorted(win_rates_all.items(), key=lambda x: x[1][0], reverse=True)
    for sport_code, (win_rate, wins, total) in sorted_all:
        display_name = SPORT_DISPLAY_NAMES.get(sport_code, sport_code.upper())
        ws_raw.write(row, 0, display_name, cell_format)
        ws_raw.write(row, 1, wins, cell_format)
        ws_raw.write(row, 2, total, cell_format)
        ws_raw.write(row, 3, round(win_rate, 1), percent_format)
        row += 1

    workbook.close()
    print(f"Excel saved to {output_file} (7 tabs)")


def main():
    """Main entry point."""
    print("=" * 50)
    print("Actionable Betting Insights Analysis")
    print("=" * 50)

    # Load all data
    df = load_data(INPUT_FILE)

    # === 2025 Data ===
    print("\n--- 2025 Data ---")
    df_2025 = filter_by_year(df, 2025)

    print("\nCalculating favourite win rates (2025):")
    win_rates_2025 = calculate_favourite_win_rates(df_2025)

    # === All Years Data ===
    print("\n--- All Years Data ---")
    print(f"Using all {len(df)} events")

    print("\nCalculating favourite win rates (All Years):")
    win_rates_all = calculate_favourite_win_rates(df)

    # === Focused Insights ===
    print("\n--- Generating Insights ---")

    print("\nCalculating market calibration...")
    calibration_data = calculate_market_calibration(df)
    print(f"  {len(calibration_data)} price buckets analyzed")

    print("\nCalculating underdog opportunities...")
    underdog_data = calculate_underdog_value(df)
    print(f"  {len(underdog_data)} sport/threshold combinations")

    print("\nCalculating sport guide...")
    sport_guide_data = calculate_sport_guide(df)
    for s in sport_guide_data:
        roi_str = f"{s['dog_roi']:+.1f}%" if s['dog_roi'] else "N/A"
        print(f"  {s['sport']}: {s['recommendation']} (ROI: {roi_str})")

    print("\nCalculating reliable favorites...")
    reliable_favorites_data = calculate_reliable_favorites(df)
    for sport, teams in reliable_favorites_data.items():
        print(f"  {sport}: {len(teams)} top teams")

    print("\nGenerating quick reference...")
    quick_ref_data = calculate_quick_reference(sport_guide_data, underdog_data)

    # Create Excel workbook with 7 tabs
    print("\n--- Generating Excel Workbook ---")
    create_excel_output(
        win_rates_2025, win_rates_all, calibration_data,
        underdog_data, sport_guide_data, reliable_favorites_data,
        quick_ref_data, EXCEL_OUTPUT_FILE
    )

    print("\nDone! Excel file has 7 focused tabs with actionable betting insights.")


if __name__ == "__main__":
    main()
