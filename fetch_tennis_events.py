#!/usr/bin/env python3
"""
Fetch tennis moneyline events from Polymarket API and save to CSV.
"""

import csv
import json
import time
from datetime import datetime

import requests


API_URL = "https://gamma-api.polymarket.com/events"
PRICES_API_URL = "https://clob.polymarket.com/prices-history"
TENNIS_TAG_ID = 864
OUTPUT_FILE = "polymarket_tennis_events.csv"
PAGE_SIZE = 100

# Time intervals before match end (in seconds)
# "open" is special - means first available price
TIME_INTERVALS = {
    "48h": 48 * 60 * 60,       # 48 hours before
    "24h": 24 * 60 * 60,       # 24 hours before
    "4h": 4 * 60 * 60,         # 4 hours before
    "1h": 60 * 60,             # 1 hour before
    "close": 5 * 60,           # 5 minutes before (closing odds)
}


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


def fetch_price_history(token_id, start_ts, end_ts):
    """Fetch price history for a token within a time range."""
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": 60,
    }
    try:
        response = requests.get(PRICES_API_URL, params=params)
        response.raise_for_status()
        return response.json().get("history", [])
    except requests.RequestException:
        return []


def find_price_at_time(history, target_ts):
    """Find the price closest to (but before or at) target timestamp."""
    if not history:
        return None

    # Filter to prices at or before target time
    valid = [p for p in history if p["t"] <= target_ts]
    if not valid:
        return None

    # Return the closest one (max timestamp)
    closest = max(valid, key=lambda x: x["t"])
    return closest["p"]


def get_opening_price(history):
    """Get the first (opening) price from history."""
    if not history:
        return None
    # Sort by timestamp and get the earliest
    sorted_history = sorted(history, key=lambda x: x["t"])
    return sorted_history[0]["p"] if sorted_history else None


def get_true_closing_price(history):
    """Get the last non-resolution price from history.

    Resolution prices are 1.0 (winner) or 0.0 (loser), which we filter out
    to get the actual pre-match closing price.
    """
    if not history:
        return None
    # Sort by timestamp descending (most recent first)
    sorted_history = sorted(history, key=lambda x: x["t"], reverse=True)
    for point in sorted_history:
        # Skip resolution prices (1.0/0.0 or very close)
        if 0.02 < point["p"] < 0.98:
            return point["p"]
    return None


def get_historical_prices_both(token_id_1, token_id_2, end_date_str):
    """Get prices for BOTH players at all intervals before match end."""
    # Build empty result with all columns
    empty_prices = {"p1_open": "", "p2_open": ""}
    for key in TIME_INTERVALS:
        empty_prices[f"p1_{key}"] = ""
        empty_prices[f"p2_{key}"] = ""

    if not end_date_str:
        return empty_prices

    try:
        # Parse end_date to timestamp
        end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        end_ts = int(end_dt.timestamp())
    except (ValueError, AttributeError):
        return empty_prices

    # Calculate start_ts (2 weeks before match to capture opening prices)
    start_ts = end_ts - (14 * 24 * 60 * 60)  # 2 weeks lookback

    # Fetch price history for both tokens
    history_1 = []
    history_2 = []

    if token_id_1:
        history_1 = fetch_price_history(token_id_1, start_ts, end_ts)
        time.sleep(0.2)

    if token_id_2:
        history_2 = fetch_price_history(token_id_2, start_ts, end_ts)
        time.sleep(0.2)

    # Build result with prices for both players
    prices = {}

    # Opening prices (first available)
    prices["p1_open"] = get_opening_price(history_1) or ""
    prices["p2_open"] = get_opening_price(history_2) or ""

    # Prices at each interval (except "close" which needs special handling)
    for key, seconds in TIME_INTERVALS.items():
        if key == "close":
            # Use true closing price (last non-resolution price)
            prices["p1_close"] = get_true_closing_price(history_1) or ""
            prices["p2_close"] = get_true_closing_price(history_2) or ""
        else:
            target_ts = end_ts - seconds
            p1 = find_price_at_time(history_1, target_ts)
            p2 = find_price_at_time(history_2, target_ts)
            prices[f"p1_{key}"] = p1 if p1 is not None else ""
            prices[f"p2_{key}"] = p2 if p2 is not None else ""

    return prices


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
    token_ids_raw = market.get("clobTokenIds", "[]")

    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
    except json.JSONDecodeError:
        outcomes = ["", ""]
        prices = ["", ""]
        token_ids = ["", ""]

    # Ensure we have 2 outcomes and prices
    outcome_1 = outcomes[0] if len(outcomes) > 0 else ""
    outcome_2 = outcomes[1] if len(outcomes) > 1 else ""
    price_1 = prices[0] if len(prices) > 0 else ""
    price_2 = prices[1] if len(prices) > 1 else ""
    token_id_1 = token_ids[0] if len(token_ids) > 0 else ""
    token_id_2 = token_ids[1] if len(token_ids) > 1 else ""

    is_closed = event.get("closed", False)
    winner = determine_winner(outcomes, prices, is_closed)

    # Get historical prices for BOTH players
    end_date = event.get("endDate", "")
    hist = get_historical_prices_both(token_id_1, token_id_2, end_date)

    return {
        "event_id": event.get("id", ""),
        "title": event.get("title", ""),
        "slug": event.get("slug", ""),
        "start_date": event.get("startDate", ""),
        "end_date": end_date,
        "active": event.get("active", False),
        "closed": is_closed,
        "liquidity": event.get("liquidity", 0),
        "volume": event.get("volume", 0),
        "outcome_1": outcome_1,
        "outcome_2": outcome_2,
        "price_1": price_1,
        "price_2": price_2,
        "winner": winner,
        "condition_id": market.get("conditionId", ""),
        "token_id_1": token_id_1,
        "token_id_2": token_id_2,
        # Historical prices for both players
        "p1_open": hist.get("p1_open", ""),
        "p2_open": hist.get("p2_open", ""),
        "p1_48h": hist.get("p1_48h", ""),
        "p2_48h": hist.get("p2_48h", ""),
        "p1_24h": hist.get("p1_24h", ""),
        "p2_24h": hist.get("p2_24h", ""),
        "p1_4h": hist.get("p1_4h", ""),
        "p2_4h": hist.get("p2_4h", ""),
        "p1_1h": hist.get("p1_1h", ""),
        "p2_1h": hist.get("p2_1h", ""),
        "p1_close": hist.get("p1_close", ""),
        "p2_close": hist.get("p2_close", ""),
    }


def save_to_csv(events, filename):
    """Save flattened moneyline events to a CSV file."""
    if not events:
        print("No events to save.")
        return

    fieldnames = [
        "event_id", "title", "slug", "start_date", "end_date",
        "active", "closed", "liquidity", "volume",
        "outcome_1", "outcome_2", "price_1", "price_2", "winner",
        "condition_id", "token_id_1", "token_id_2",
        # Historical prices for both players at each interval
        "p1_open", "p2_open",
        "p1_48h", "p2_48h",
        "p1_24h", "p2_24h",
        "p1_4h", "p2_4h",
        "p1_1h", "p2_1h",
        "p1_close", "p2_close",
    ]

    # Flatten all events (with progress indicator for price fetching)
    print(f"Fetching historical prices for {len(events)} events...")
    flattened = []
    for i, event in enumerate(events):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing event {i + 1}/{len(events)}...")
        flattened.append(flatten_event(event))

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
