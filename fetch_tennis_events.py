#!/usr/bin/env python3
"""
Fetch tennis moneyline events from Polymarket API and save to CSV.
"""

import csv
import json
import time
import requests


API_URL = "https://gamma-api.polymarket.com/events"
TENNIS_TAG_ID = 864
OUTPUT_FILE = "polymarket_tennis_events.csv"
PAGE_SIZE = 100


def fetch_tennis_events_page(offset=0):
    """Fetch a single page of tennis events."""
    params = {
        "tag_id": TENNIS_TAG_ID,
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    return response.json()


def fetch_all_tennis_events():
    """Fetch all tennis events with pagination."""
    all_events = []
    offset = 0

    while True:
        print(f"Fetching events (offset={offset})...")
        events = fetch_tennis_events_page(offset)

        if not events:
            break

        all_events.extend(events)
        offset += PAGE_SIZE

        # Small delay to avoid rate limiting
        time.sleep(0.2)

    return all_events


def is_moneyline_event(event):
    """Check if event is a moneyline (head-to-head) event."""
    markets = event.get("markets", [])

    # Must have exactly 1 market
    if len(markets) != 1:
        return False

    market = markets[0]
    outcomes_raw = market.get("outcomes", "[]")

    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    except json.JSONDecodeError:
        return False

    # Must have exactly 2 outcomes
    if len(outcomes) != 2:
        return False

    # Must NOT be Yes/No (those are tournament winner markets)
    if "Yes" in outcomes or "No" in outcomes:
        return False

    return True


def determine_winner(outcomes, prices, is_closed):
    """Determine the winner based on prices (winner has price=1)."""
    if not is_closed:
        return ""

    for i, price in enumerate(prices):
        if price == "1" or price == 1:
            return outcomes[i] if i < len(outcomes) else ""

    return ""


def flatten_event(event):
    """Extract essential fields from a moneyline event."""
    markets = event.get("markets", [])
    market = markets[0] if markets else {}

    # Parse outcomes and prices
    outcomes_raw = market.get("outcomes", "[]")
    prices_raw = market.get("outcomePrices", "[]")

    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    except json.JSONDecodeError:
        outcomes = ["", ""]
        prices = ["", ""]

    # Ensure we have 2 outcomes and prices
    outcome_1 = outcomes[0] if len(outcomes) > 0 else ""
    outcome_2 = outcomes[1] if len(outcomes) > 1 else ""
    price_1 = prices[0] if len(prices) > 0 else ""
    price_2 = prices[1] if len(prices) > 1 else ""

    is_closed = event.get("closed", False)
    winner = determine_winner(outcomes, prices, is_closed)

    return {
        "event_id": event.get("id", ""),
        "title": event.get("title", ""),
        "slug": event.get("slug", ""),
        "start_date": event.get("startDate", ""),
        "end_date": event.get("endDate", ""),
        "active": event.get("active", False),
        "closed": is_closed,
        "liquidity": event.get("liquidity", 0),
        "volume": event.get("volume", 0),
        "outcome_1": outcome_1,
        "outcome_2": outcome_2,
        "price_1": price_1,
        "price_2": price_2,
        "winner": winner,
    }


def save_to_csv(events, filename):
    """Save flattened moneyline events to a CSV file."""
    if not events:
        print("No events to save.")
        return

    fieldnames = [
        "event_id", "title", "slug", "start_date", "end_date",
        "active", "closed", "liquidity", "volume",
        "outcome_1", "outcome_2", "price_1", "price_2", "winner"
    ]

    # Flatten all events
    flattened = [flatten_event(event) for event in events]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)

    print(f"Saved {len(flattened)} moneyline events to {filename}")


def main():
    """Main entry point."""
    try:
        print(f"Fetching tennis events from {API_URL}...")
        all_events = fetch_all_tennis_events()
        print(f"Found {len(all_events)} total tennis events.")

        # Filter to moneyline events only
        moneyline_events = [e for e in all_events if is_moneyline_event(e)]
        print(f"Filtered to {len(moneyline_events)} moneyline events.")

        save_to_csv(moneyline_events, OUTPUT_FILE)
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
