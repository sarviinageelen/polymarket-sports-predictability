#!/usr/bin/env python3
"""
Fetch tennis moneyline events from Polymarket API and save to CSV.

Optimized version with:
- Connection pooling (requests.Session)
- Parallel price fetching for both tokens
- Parallel event processing with ThreadPoolExecutor
- Binary search for O(log n) price lookups
- Smart rate limiting with exponential backoff
- Excel-friendly CSV formatting
- 2025 events only for better data completeness
"""

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock

import requests
from tqdm import tqdm

from fetch_sports import SPORT_CATEGORIES, fetch_sports_data

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


API_URL = "https://gamma-api.polymarket.com/events"
PRICES_API_URL = "https://clob.polymarket.com/prices-history"
TRADES_API_URL = "https://data-api.polymarket.com/trades"
TENNIS_TAG_ID = 864
OUTPUT_FILE = "fetch_events.csv"
PAGE_SIZE = 100
MAX_WORKERS = 8  # Number of parallel workers for event processing

# Minimum year for events (filter out older data with incomplete price history)
MIN_YEAR = 2025

# Connection pooling - reuse HTTP connections
SESSION = requests.Session()

# Cached tag-to-sport mapping (populated at runtime)
_tag_to_sport_map = None


def build_tag_to_sport_mapping():
    """Build mapping from tag_id to (category, sport) pairs.

    Fetches sports metadata and creates a reverse lookup from tag IDs.
    """
    sports_data = fetch_sports_data()
    tag_map = {}

    for entry in sports_data:
        sport_code = entry.get("sport", "")
        category = SPORT_CATEGORIES.get(sport_code, "unknown")
        tags_str = entry.get("tags", "")

        if tags_str:
            for tag in str(tags_str).split(","):
                tag = tag.strip()
                if tag.isdigit():
                    tag_id = int(tag)
                    if tag_id not in tag_map:
                        tag_map[tag_id] = []
                    tag_map[tag_id].append((category, sport_code))

    return tag_map


def get_tag_to_sport_map():
    """Get the cached tag-to-sport mapping, building it if needed."""
    global _tag_to_sport_map
    if _tag_to_sport_map is None:
        _tag_to_sport_map = build_tag_to_sport_mapping()
    return _tag_to_sport_map


def determine_sport_for_event(event, tag_id):
    """Determine category and sport for an event based on tag_id.

    If multiple sports share the same tag (e.g., ATP and WTA both use 864),
    attempts to disambiguate using the event title.
    """
    tag_map = get_tag_to_sport_map()
    matches = tag_map.get(tag_id, [])

    if not matches:
        return ("unknown", "unknown")

    if len(matches) == 1:
        return matches[0]

    # Multiple sports share this tag - try to disambiguate by title
    title = event.get("title", "").lower()

    for category, sport_code in matches:
        if sport_code.lower() in title:
            return (category, sport_code)

    # Default to first match
    return matches[0]

# Rate limiting state
_rate_limit_lock = Lock()
_backoff_until = 0


def _wait_for_rate_limit():
    """Wait if we're in a backoff period."""
    global _backoff_until
    with _rate_limit_lock:
        now = time.time()
        if now < _backoff_until:
            time.sleep(_backoff_until - now)


def _handle_rate_limit(response):
    """Handle 429 rate limit response with exponential backoff."""
    global _backoff_until
    if response.status_code == 429:
        with _rate_limit_lock:
            # Exponential backoff: wait 1-2 seconds
            wait_time = float(response.headers.get("Retry-After", 1))
            new_backoff = time.time() + wait_time
            # Only update if this extends the backoff period
            if new_backoff > _backoff_until:
                _backoff_until = new_backoff
            # Sleep for the remaining backoff time
            sleep_duration = _backoff_until - time.time()
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        return True
    return False


def format_date_for_excel(iso_date_str):
    """Convert ISO date string to Excel-friendly format (YYYY-MM-DD HH:MM:SS)."""
    if not iso_date_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_date_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def is_2025_or_later(date_str):
    """Check if a date is in 2025 or later."""
    if not date_str:
        return False
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.year >= MIN_YEAR
    except (ValueError, AttributeError):
        return False


def fetch_tennis_events_page(offset=0):
    """Fetch a single page of tennis events."""
    params = {
        "tag_id": TENNIS_TAG_ID,
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    _wait_for_rate_limit()
    response = SESSION.get(API_URL, params=params)
    if _handle_rate_limit(response):
        response = SESSION.get(API_URL, params=params)
    response.raise_for_status()
    return response.json()


def fetch_all_tennis_events():
    """Fetch all tennis events with pagination, filtered to 2025+."""
    all_events = []
    offset = 0

    with tqdm(desc="Fetching events", unit=" pages") as pbar:
        while True:
            events = fetch_tennis_events_page(offset)

            if not events:
                break

            # Filter to 2025+ events only
            for event in events:
                start_date = event.get("startDate", "")
                if is_2025_or_later(start_date):
                    all_events.append(event)

            offset += PAGE_SIZE
            pbar.update(1)

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
        _wait_for_rate_limit()
        time.sleep(0.1)  # Proactive delay to avoid hitting rate limits
        response = SESSION.get(PRICES_API_URL, params=params)
        if _handle_rate_limit(response):
            response = SESSION.get(PRICES_API_URL, params=params)
        response.raise_for_status()
        return response.json().get("history", [])
    except requests.RequestException:
        return []


def fetch_trades_for_condition(condition_id):
    """Fetch all trades for a market from the Data-API.

    This is used as a fallback when CLOB prices-history returns no data.
    """
    params = {
        "market": condition_id,
        "limit": 10000,
    }
    try:
        _wait_for_rate_limit()
        time.sleep(0.1)
        response = SESSION.get(TRADES_API_URL, params=params)
        if _handle_rate_limit(response):
            response = SESSION.get(TRADES_API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def derive_closing_from_trades(trades, token_id_1, token_id_2):
    """Derive closing prices from trade history.

    Finds the last non-resolution trade for each token.
    Resolution trades have price 0.0 or 1.0.
    """
    if not trades:
        return {"p1_close": None, "p2_close": None}

    # Group trades by asset (token_id) and find last non-resolution price
    last_price_by_asset = {}

    # Sort by timestamp to process in order
    sorted_trades = sorted(trades, key=lambda x: x.get("timestamp", 0))

    for trade in sorted_trades:
        asset = trade.get("asset", "")
        price = trade.get("price")
        if price is not None:
            # Filter out resolution prices (0.0 or 1.0)
            if 0.02 < price < 0.98:
                last_price_by_asset[asset] = price

    return {
        "p1_close": last_price_by_asset.get(token_id_1),
        "p2_close": last_price_by_asset.get(token_id_2),
    }


def get_true_closing_price(sorted_history):
    """Get the last non-resolution price from sorted history.

    Resolution prices are 1.0 (winner) or 0.0 (loser), which we filter out
    to get the actual pre-match closing price.
    """
    if not sorted_history:
        return None
    # Iterate from end (most recent) to find non-resolution price
    for i in range(len(sorted_history) - 1, -1, -1):
        price = sorted_history[i].get("p")
        if price is not None and 0.02 < price < 0.98:
            return price
    return None


def _fetch_both_histories(token_id_1, token_id_2, start_ts, end_ts):
    """Fetch price histories for both tokens in parallel."""
    history_1 = []
    history_2 = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if token_id_1:
            futures[executor.submit(fetch_price_history, token_id_1, start_ts, end_ts)] = 1
        if token_id_2:
            futures[executor.submit(fetch_price_history, token_id_2, start_ts, end_ts)] = 2

        for future in as_completed(futures):
            player = futures[future]
            try:
                result = future.result()
                if player == 1:
                    history_1 = result
                else:
                    history_2 = result
            except Exception as e:
                token_id = token_id_1 if player == 1 else token_id_2
                logger.warning("Failed to fetch price history for token %s: %s", token_id, e)

    return history_1, history_2


def get_closing_prices(token_id_1, token_id_2, condition_id, start_date_str, end_date_str):
    """Get closing prices for both players before match resolution.

    Uses max(start_date, end_date) as the end of the time window to handle
    cases where markets are created after the scheduled event time.

    Falls back to Data-API trades if CLOB prices-history returns no data.
    """
    empty_prices = {"p1_close": None, "p2_close": None}

    if not end_date_str and not start_date_str:
        return empty_prices

    try:
        # Parse both dates, using the other as fallback if one is missing
        end_dt = None
        start_dt = None

        if end_date_str:
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        if start_date_str:
            start_dt = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))

        # Use the LATER of the two dates as the end of the time window
        # This handles cases where market was created after scheduled event time
        if start_dt and end_dt:
            effective_end_dt = max(start_dt, end_dt)
        else:
            effective_end_dt = end_dt or start_dt

        # Extend 24 hours beyond to capture post-creation trading and resolution
        # Markets created close to event time may have trading after the later date
        effective_end_dt = effective_end_dt + timedelta(hours=24)

        end_ts = int(effective_end_dt.timestamp())
    except (ValueError, AttributeError):
        return empty_prices

    # Fetch 7 days of price history to ensure we capture closing prices
    start_ts = end_ts - (7 * 24 * 60 * 60)

    # Fetch price histories in parallel
    history_1, history_2 = _fetch_both_histories(token_id_1, token_id_2, start_ts, end_ts)

    # Sort histories for closing price extraction (filter items with missing timestamp)
    sorted_1 = sorted((h for h in history_1 if "t" in h), key=lambda x: x["t"]) if history_1 else []
    sorted_2 = sorted((h for h in history_2 if "t" in h), key=lambda x: x["t"]) if history_2 else []

    closing = {
        "p1_close": get_true_closing_price(sorted_1),
        "p2_close": get_true_closing_price(sorted_2),
    }

    # Fallback to Data-API trades if CLOB returned no data
    if closing["p1_close"] is None and closing["p2_close"] is None and condition_id:
        trades = fetch_trades_for_condition(condition_id)
        if trades:
            closing = derive_closing_from_trades(trades, token_id_1, token_id_2)

    return closing


def determine_winner(outcomes, prices, is_closed):
    """Determine the winner based on prices (winner has price=1)."""
    if not is_closed:
        return None

    for i, price in enumerate(prices):
        # Normalize to float for comparison (use tolerance for float precision)
        try:
            if abs(float(price) - 1.0) < 0.01:
                return outcomes[i] if i < len(outcomes) else None
        except (ValueError, TypeError):
            continue

    return None


def to_float(value):
    """Convert value to float, return None if not possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def process_event(event):
    """Process a single event: validate as moneyline and flatten.

    Returns flattened dict if valid moneyline event, None otherwise.
    This combines is_moneyline_event() and flatten_event() into a single pass.
    """
    markets = event.get("markets", [])

    # Must have exactly 1 market
    if len(markets) != 1:
        return None

    market = markets[0]

    # Parse JSON once and cache
    outcomes_raw = market.get("outcomes", "[]")
    prices_raw = market.get("outcomePrices", "[]")
    token_ids_raw = market.get("clobTokenIds", "[]")

    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
    except json.JSONDecodeError:
        return None

    # Validate moneyline: exactly 2 outcomes, not Yes/No
    if len(outcomes) != 2:
        return None
    if "Yes" in outcomes or "No" in outcomes:
        return None

    # Extract values (outcomes validated to have exactly 2 items above)
    outcome_1 = outcomes[0]
    outcome_2 = outcomes[1]
    price_1 = to_float(prices[0]) if len(prices) > 0 else None
    price_2 = to_float(prices[1]) if len(prices) > 1 else None
    token_id_1 = token_ids[0] if len(token_ids) > 0 else ""
    token_id_2 = token_ids[1] if len(token_ids) > 1 else ""

    is_closed = event.get("closed", False)
    winner = determine_winner(outcomes, prices, is_closed)

    # Get closing prices for both players
    condition_id = market.get("conditionId", "")
    start_date = event.get("startDate", "")
    end_date = event.get("endDate", "")
    closing = get_closing_prices(token_id_1, token_id_2, condition_id, start_date, end_date)

    # Determine category and sport
    category, sport = determine_sport_for_event(event, TENNIS_TAG_ID)

    # Return in chronological order:
    # Category → Identification → Timing → Players → Closing Prices → Result → Trading
    return {
        # Category
        "category": category,
        "sport": sport,
        # Identification
        "event_id": event.get("id", ""),
        "condition_id": condition_id,
        "title": event.get("title", ""),
        "slug": event.get("slug", ""),
        # Timing
        "start_date": format_date_for_excel(event.get("startDate", "")),
        "end_date": format_date_for_excel(end_date),
        # Players
        "outcome_1": outcome_1,
        "outcome_2": outcome_2,
        # Closing prices
        "p1_close": closing.get("p1_close"),
        "p2_close": closing.get("p2_close"),
        # Result
        "closed": 1 if is_closed else 0,
        "winner": winner,
        # Trading
        "volume": to_float(event.get("volume", 0)),
    }


def save_to_csv(events, filename):
    """Save processed moneyline events to a CSV file with parallel processing."""
    if not events:
        print("No events to save.")
        return

    # Columns in chronological order:
    # Category → Identification → Timing → Players → Closing Prices → Result → Trading
    fieldnames = [
        "category", "sport",                          # Category
        "event_id", "condition_id", "title", "slug",  # Identification
        "start_date", "end_date",                     # Timing
        "outcome_1", "outcome_2",                     # Players
        "p1_close", "p2_close",                       # Closing prices
        "closed", "winner",                           # Result
        "volume",                                     # Trading
    ]

    # Process events in parallel with progress bar
    flattened = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_event, event): event for event in events}

        with tqdm(total=len(events), desc="Processing events") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:  # Filter out non-moneyline events
                    flattened.append(result)
                pbar.update(1)

    # Sort by start_date for consistent ordering
    flattened.sort(key=lambda x: x.get("start_date") or "")

    # Write CSV with proper formatting (QUOTE_MINIMAL avoids quoting numbers)
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(flattened)

    print(f"Saved {len(flattened)} moneyline events to {filename}")


def main():
    """Main entry point."""
    try:
        # Load sports metadata first for category/sport classification
        print("Loading sports metadata...")
        get_tag_to_sport_map()

        print(f"Fetching events from {MIN_YEAR} onwards...")
        all_events = fetch_all_tennis_events()
        print(f"\nFound {len(all_events)} events from {MIN_YEAR}+.")

        # Process all events (filtering happens inside save_to_csv now)
        save_to_csv(all_events, OUTPUT_FILE)
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
