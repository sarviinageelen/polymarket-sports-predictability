#!/usr/bin/env python3
"""
Generate a "How often do favourites win?" chart from Polymarket data.

Creates a horizontal bar chart showing favourite win rates by sport,
styled similar to The Athletic's sports analytics charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Input/Output files
INPUT_FILE = "fetch_events.csv"
OUTPUT_FILE = "favourite_win_rates.png"

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

# Chart styling
BAR_COLOR = "#3B8686"  # Teal color matching The Athletic's style
BACKGROUND_COLOR = "#F5F5F0"  # Light cream background
TEXT_COLOR = "#333333"  # Dark gray for text
GRID_COLOR = "#E0E0E0"  # Light gray for grid


def load_data(filename):
    """Load event data from CSV file."""
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} events from {filename}")
    return df


def calculate_favourite_win_rates(df):
    """Calculate favourite win rate for each sport.

    The favourite is the outcome with the higher closing price.
    Win rate = (times favourite won / total closed events) * 100%

    Returns a dict: {sport_code: (win_rate, wins, total)}
    """
    # Filter to closed events only (where we have a winner)
    closed_df = df[df["closed"] == 1].copy()
    print(f"Analyzing {len(closed_df)} closed events")

    # Merge ncaab and cbb into single "cbb" category
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

            # Skip rows with missing data
            if pd.isna(p1_close) or pd.isna(p2_close) or pd.isna(winner):
                continue

            # Identify the favourite (higher closing price)
            if p1_close > p2_close:
                favourite = outcome_1
            elif p2_close > p1_close:
                favourite = outcome_2
            else:
                # Equal prices - skip (no clear favourite)
                continue

            # Check if favourite won
            if favourite == winner:
                favourite_wins += 1

            total += 1

        if total > 0:
            win_rate = (favourite_wins / total) * 100
            results[sport] = (win_rate, favourite_wins, total)
            print(f"  {sport.upper()}: {win_rate:.1f}% ({favourite_wins}/{total})")

    return results


def create_chart(data, output_file):
    """Create a horizontal bar chart of favourite win rates.

    Args:
        data: dict of {sport_code: (win_rate, wins, total)}
        output_file: path to save the PNG chart
    """
    # Prepare data for plotting
    sports = []
    win_rates = []
    sample_sizes = []

    # Sort by win rate descending
    sorted_data = sorted(data.items(), key=lambda x: x[1][0], reverse=True)

    for sport_code, (win_rate, wins, total) in sorted_data:
        display_name = SPORT_DISPLAY_NAMES.get(sport_code, sport_code.upper())
        sports.append(display_name)
        win_rates.append(win_rate)
        sample_sizes.append((sport_code, total))

    # Reverse lists so highest appears at top (matplotlib plots index 0 at bottom)
    sports.reverse()
    win_rates.reverse()
    sample_sizes.reverse()

    # Create figure with 16:9 widescreen format (1920x1080 at 150 dpi)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    # Create horizontal bars with better spacing
    y_pos = range(len(sports))
    bars = ax.barh(y_pos, win_rates, color=BAR_COLOR, height=0.65)

    # Add percentage labels at end of bars (outside for visibility)
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        ax.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.0f}%",
            va="center", ha="left",
            fontsize=20, fontweight="bold",
            color=TEXT_COLOR
        )

    # Set sport names on y-axis (left-aligned, larger font)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sports, fontsize=18, ha="right")

    # Set x-axis range and labels
    ax.set_xlim(0, 110)  # Extra space for labels
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%"], fontsize=14, color="#888888")

    # Add light vertical grid lines
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove spines
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Remove y-axis ticks
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    # Add title and subtitle (reduced size)
    fig.text(
        0.06, 0.93,
        "How often do favourites win?",
        fontsize=28, fontweight="bold",
        color=TEXT_COLOR, ha="left"
    )
    fig.text(
        0.06, 0.88,
        "Average favourite win rates by sport",
        fontsize=15, color="#777777", ha="left"
    )

    # Build sample size footer (single line)
    sample_parts = []
    for code, total in sample_sizes:
        sport_name = SPORT_DISPLAY_NAMES.get(code, code.upper())
        sample_parts.append(f"{sport_name}: {total:,}")

    sample_text = "  |  ".join(sample_parts)

    # Add footer notes
    fig.text(
        0.06, 0.08,
        "Favourites calculated using Polymarket closing prices",
        fontsize=11, color="#999999", ha="left", style="italic"
    )
    fig.text(
        0.06, 0.04,
        f"Sample sizes: {sample_text}",
        fontsize=10, color="#999999", ha="left"
    )

    # Adjust layout
    plt.subplots_adjust(left=0.18, right=0.90, top=0.84, bottom=0.12)

    # Save chart
    plt.savefig(output_file, dpi=150, facecolor=BACKGROUND_COLOR, bbox_inches="tight")
    plt.close()

    print(f"\nChart saved to {output_file}")


def main():
    """Main entry point."""
    print("=" * 50)
    print("Favourite Win Rate Analysis")
    print("=" * 50)

    # Load data
    df = load_data(INPUT_FILE)

    # Calculate win rates
    print("\nCalculating favourite win rates by sport:")
    win_rates = calculate_favourite_win_rates(df)

    # Create chart
    print("\nGenerating chart...")
    create_chart(win_rates, OUTPUT_FILE)

    print("\nDone!")


if __name__ == "__main__":
    main()
