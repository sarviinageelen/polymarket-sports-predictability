#!/usr/bin/env python3
"""
Fetch sports metadata from Polymarket API and save to CSV.
"""

import csv

import requests

# Connection pooling - reuse HTTP connections
SESSION = requests.Session()

API_URL = "https://gamma-api.polymarket.com/sports"
OUTPUT_FILE = "../data/fetch_sports.csv"

# Sport category mapping (lowercase, database-friendly)
SPORT_CATEGORIES = {
    # soccer
    "epl": "soccer", "lal": "soccer", "bun": "soccer", "fl1": "soccer",
    "sea": "soccer", "ucl": "soccer", "uel": "soccer", "col": "soccer",
    "afc": "soccer", "ofc": "soccer", "fif": "soccer", "ere": "soccer",
    "arg": "soccer", "itc": "soccer", "mex": "soccer", "lcs": "soccer",
    "lib": "soccer", "sud": "soccer", "tur": "soccer", "con": "soccer",
    "cof": "soccer", "uef": "soccer", "caf": "soccer", "rus": "soccer",
    "efa": "soccer", "efl": "soccer", "mls": "soccer", "cdr": "soccer",
    "cde": "soccer", "dfb": "soccer", "bra": "soccer", "jap": "soccer",
    "ja2": "soccer", "kor": "soccer", "spl": "soccer", "chi": "soccer",
    "aus": "soccer", "ind": "soccer", "nor": "soccer", "den": "soccer",
    "por": "soccer",
    # basketball
    "ncaab": "basketball", "wnba": "basketball", "nba": "basketball",
    "cwbb": "basketball", "cbb": "basketball",
    # american_football
    "nfl": "american_football", "cfb": "american_football",
    # baseball
    "mlb": "baseball", "kbo": "baseball",
    # ice_hockey
    "nhl": "ice_hockey", "shl": "ice_hockey", "cehl": "ice_hockey",
    "dehl": "ice_hockey", "snhl": "ice_hockey", "khl": "ice_hockey",
    "ahl": "ice_hockey",
    # cricket
    "ipl": "cricket", "odi": "cricket", "t20": "cricket", "abb": "cricket",
    "csa": "cricket", "test": "cricket", "she": "cricket", "sasa": "cricket",
    "lpl": "cricket", "psp": "cricket", "crint": "cricket", "craus": "cricket",
    "creng": "cricket", "crnew": "cricket", "crind": "cricket",
    "crsou": "cricket", "crpak": "cricket", "cruae": "cricket",
    # tennis
    "atp": "tennis", "wta": "tennis",
    # mma
    "mma": "mma",
    # esports
    "dota2": "esports", "lol": "esports", "val": "esports", "cs2": "esports",
    "mlbb": "esports", "ow": "esports", "codmw": "esports", "fifa": "esports",
    "pubg": "esports", "r6siege": "esports", "rl": "esports", "hok": "esports",
    "wildrift": "esports", "sc2": "esports", "sc": "esports",
}


def get_category(sport_code):
    """Get the category for a sport code."""
    return SPORT_CATEGORIES.get(sport_code, "unknown")


def fetch_sports_data():
    """Fetch sports metadata from the Polymarket API."""
    response = SESSION.get(API_URL)
    response.raise_for_status()
    return response.json()


def save_to_csv(data, filename):
    """Save sports data to a CSV file with category column."""
    if not data:
        print("No data to save.")
        return

    fieldnames = ["category", "sport", "image", "resolution", "ordering", "tags", "series"]
    unmapped_sports = []
    processed_data = []

    # Create new rows with category (avoid mutating original data)
    for row in data:
        sport_code = row.get("sport", "")
        category = get_category(sport_code)
        new_row = {**row, "category": category}
        processed_data.append(new_row)
        if category == "unknown" and sport_code:
            unmapped_sports.append(sport_code)

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(processed_data)

    print(f"Saved {len(data)} records to {filename}")

    if unmapped_sports:
        print(f"\nWARNING: {len(unmapped_sports)} unmapped sport(s) found:")
        for sport in unmapped_sports:
            print(f"  - {sport}")
        print("Consider adding these to SPORT_CATEGORIES mapping.")


def main():
    """Main entry point."""
    try:
        print(f"Fetching data from {API_URL}...")
        data = fetch_sports_data()
        save_to_csv(data, OUTPUT_FILE)
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
