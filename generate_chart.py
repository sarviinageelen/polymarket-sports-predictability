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
    "cfb": "College Football",
    "ncaab": "College Basketball",
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

    results = {}

    for sport in df["sport"].unique():
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

    # Sort by win rate descending
    sorted_data = sorted(data.items(), key=lambda x: x[1][0], reverse=True)

    for sport_code, (win_rate, wins, total) in sorted_data:
        display_name = SPORT_DISPLAY_NAMES.get(sport_code, sport_code.upper())
        sports.append(display_name)
        win_rates.append(win_rate)

    # Create figure with The Athletic's style
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    # Create horizontal bars (reversed so highest is at top)
    y_pos = range(len(sports))
    bars = ax.barh(y_pos, win_rates, color=BAR_COLOR, height=0.7)

    # Add percentage labels inside bars
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        # Position label inside the bar, near the right edge
        label_x = bar.get_width() - 3 if bar.get_width() > 15 else bar.get_width() + 1
        text_color = "white" if bar.get_width() > 15 else TEXT_COLOR
        ha = "right" if bar.get_width() > 15 else "left"

        ax.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            f"{rate:.0f}%",
            va="center", ha=ha,
            fontsize=14, fontweight="bold",
            color=text_color
        )

    # Set sport names on y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sports, fontsize=13)

    # Set x-axis range and labels
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%"], fontsize=11, color="#666666")

    # Add light vertical grid lines
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove spines
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Remove y-axis ticks
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    # Add title and subtitle
    fig.text(
        0.12, 0.95,
        "How often do favourites win?",
        fontsize=22, fontweight="bold",
        color=TEXT_COLOR, ha="left"
    )
    fig.text(
        0.12, 0.90,
        "Average favourite win rates by sport",
        fontsize=13, color="#666666", ha="left"
    )

    # Add footer note
    fig.text(
        0.12, 0.02,
        "Favourites calculated using Polymarket closing prices",
        fontsize=9, color="#999999", ha="left", style="italic"
    )

    # Adjust layout
    plt.subplots_adjust(left=0.25, right=0.95, top=0.85, bottom=0.08)

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
