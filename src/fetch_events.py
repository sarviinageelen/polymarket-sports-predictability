#!/usr/bin/env python3
"""
Fetch sports moneyline events from Polymarket API and save to CSV.

Supports multiple sports: ATP, WTA, NBA, NFL, MLB, CFB, NCAAB

Optimized version with:
- Async/await with aiohttp for concurrent CLOB API price fetching
- Connection pooling for gamma-api requests
- Smart rate limiting with exponential backoff
- Excel-friendly CSV formatting
- 2025 events only for better data completeness
"""

import asyncio
import csv
import json
import logging
import time
from datetime import datetime, timedelta
from threading import Lock

import aiohttp
import requests
from tqdm import tqdm

from fetch_sports import SPORT_CATEGORIES, fetch_sports_data

# Configure logging - changed to INFO to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


API_URL = "https://gamma-api.polymarket.com/events"
PRICES_API_URL = "https://clob.polymarket.com/prices-history"
OUTPUT_FILE = "../data/fetch_events.csv"
PAGE_SIZE = 100
MAX_WORKERS = 8  # Number of concurrent async tasks for event processing

# CLOB API configuration for reliable token IDs and market data
CLOB_HOST = "https://clob.polymarket.com"
CLOB_CHAIN_ID = 137

# Sports to fetch: (tag_id, sport_codes that use this tag)
# ATP and WTA share tag 864 - disambiguation happens via title matching
SPORTS_TO_FETCH = [
    (864, ["atp", "wta"]),      # Tennis (ATP & WTA share tag)
    (745, ["nba"]),             # NBA
    (450, ["nfl"]),             # NFL
    (100381, ["mlb"]),          # MLB
    (100351, ["cfb"]),          # College Football
    (100149, ["ncaab"]),        # College Basketball (March Madness)
    (101178, ["cbb"]),          # College Basketball (Regular Season)
]

# Minimum year for events (filter out older data with incomplete price history)
MIN_YEAR = 2025

# Connection pooling for gamma-api requests
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


async def fetch_clob_markets_async():
    """Fetch all markets from CLOB API with pagination.

    Returns dict mapping condition_id -> market data with reliable token IDs,
    game timing, and winner information.

    Returns:
        dict: {condition_id: {token_id_1, token_id_2, game_start_time, winner_index, ...}}
    """
    from py_clob_client.client import ClobClient
    from functools import partial

    client = ClobClient(host=CLOB_HOST, chain_id=CLOB_CHAIN_ID)
    markets_map = {}
    next_cursor = ""
    page_count = 0

    with tqdm(desc="Fetching CLOB markets", unit=" pages", leave=False) as pbar:
        while True:
            # Run synchronous get_markets() in thread pool for async compatibility
            loop = asyncio.get_event_loop()
            try:
                if next_cursor == "":
                    response = await loop.run_in_executor(None, client.get_markets)
                else:
                    response = await loop.run_in_executor(
                        None,
                        partial(client.get_markets, next_cursor=next_cursor)
                    )
            except Exception as e:
                logger.error(f"Failed to fetch CLOB markets page {page_count}: {e}")
                break

            # Parse response
            markets = response.get("data", [])
            if not markets:
                break

            # Build condition_id mapping
            for market in markets:
                condition_id = market.get("condition_id")
                if not condition_id:
                    continue

                # Filter to active, non-archived markets only
                if market.get("archived", False):
                    continue
                if not market.get("active", True):
                    continue

                tokens = market.get("tokens", [])
                if len(tokens) != 2:
                    continue  # Only moneyline markets (2 outcomes)

                # Extract winner (if resolved)
                winner_index = None
                if tokens[0].get("winner") is True:
                    winner_index = 0
                elif tokens[1].get("winner") is True:
                    winner_index = 1

                markets_map[condition_id] = {
                    "token_id_1": tokens[0].get("token_id", ""),
                    "token_id_2": tokens[1].get("token_id", ""),
                    "game_start_time": market.get("game_start_time"),
                    "end_date_iso": market.get("end_date_iso"),  # Fallback timing
                    "winner_index": winner_index,
                    "closed": market.get("closed", False),
                    "outcome_1": tokens[0].get("outcome", ""),
                    "outcome_2": tokens[1].get("outcome", ""),
                    "is_50_50_outcome": market.get("is_50_50_outcome", False),
                    "tags": market.get("tags", []),
                    "question": market.get("question", ""),
                }

            # Pagination
            next_cursor = response.get("next_cursor", "")
            page_count += 1
            pbar.update(1)

            # Check for end of pagination
            if next_cursor == "LTE=" or not next_cursor:
                break

    logger.info(f"Fetched {len(markets_map)} CLOB markets across {page_count} pages")
    return markets_map


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


def fetch_events_page(tag_id, offset=0):
    """Fetch a single page of events for a given tag_id."""
    params = {
        "tag_id": tag_id,
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    _wait_for_rate_limit()
    response = SESSION.get(API_URL, params=params)
    if _handle_rate_limit(response):
        response = SESSION.get(API_URL, params=params)
    response.raise_for_status()
    return response.json()


def fetch_all_events(tag_id):
    """Fetch all events for a tag_id with pagination, filtered to 2025+."""
    all_events = []
    offset = 0

    with tqdm(desc="Fetching events", unit=" pages", leave=False) as pbar:
        while True:
            events = fetch_events_page(tag_id, offset)

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


async def fetch_price_history(session, token_id, start_ts, end_ts):
    """Fetch price history for a token using aiohttp."""
    if not token_id:
        logger.debug(f"Skipping price fetch - empty token_id")
        return []
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": 5,
    }
    try:
        async with session.get(PRICES_API_URL, params=params) as response:
            if response.status == 200:
                data = await response.json()
                history = data.get("history", [])
                logger.debug(f"Fetched {len(history)} price points for token {token_id[:8]}...")
                return history
            else:
                logger.warning(f"Price API returned {response.status} for token {token_id[:8]}...")
                return []
    except Exception as e:
        logger.error(f"Price fetch exception for token {token_id[:8]}...: {e}")
        return []


async def fetch_price_history_with_retry(session, token_id, start_ts, end_ts, max_retries=3):
    """Fetch price history with exponential backoff retry.

    Retries up to max_retries times with exponential backoff (1s, 2s, 4s).
    """
    for attempt in range(max_retries):
        try:
            result = await fetch_price_history(session, token_id, start_ts, end_ts)
            if result:  # Success
                return result

            # Empty result but no exception - may be transient
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.debug(f"Empty result for token {token_id[:8]}..., retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                logger.warning(f"Failed to fetch prices for token {token_id[:8]}... after {max_retries} attempts")
                return []

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Price fetch error for token {token_id[:8]}...: {e}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to fetch prices for token {token_id[:8]}... after {max_retries} attempts: {e}")
                return []

    return []



def get_price_at_match_start(sorted_history, match_start_ts):
    """Get the price at or just before match start time.

    This is the true pre-match closing price - the last price before the
    game started, when live betting effects haven't yet moved the odds.

    Args:
        sorted_history: List of {t: timestamp, p: price} sorted by timestamp
        match_start_ts: Unix timestamp of when the match started

    Returns:
        The price closest to (but not after) match start, or None if not found.
    """
    if not sorted_history:
        return None

    # Find the last price entry at or before match start
    best_price = None
    for entry in sorted_history:
        ts = entry.get("t")
        price = entry.get("p")
        if ts is None or price is None:
            continue
        if ts <= match_start_ts:
            best_price = price
        else:
            # Past match start, stop looking
            break

    return best_price


async def _fetch_both_histories(session, token_id_1, token_id_2, start_ts, end_ts):
    """Fetch price histories for both tokens in parallel using asyncio with retry logic."""
    history_1, history_2 = await asyncio.gather(
        fetch_price_history_with_retry(session, token_id_1, start_ts, end_ts),
        fetch_price_history_with_retry(session, token_id_2, start_ts, end_ts),
    )
    return history_1, history_2


async def get_closing_prices(session, token_id_1, token_id_2, condition_id, game_start_time_str):
    """Get closing prices for both players at game start time.

    Uses CLOB game_start_time as the match time (accurate timing from CLOB API).

    Args:
        token_id_1: Token ID from CLOB (reliable)
        token_id_2: Token ID from CLOB (reliable)
        condition_id: Condition ID for logging
        game_start_time_str: ISO timestamp of game start from CLOB
    """
    empty_prices = {"p1_close": None, "p2_close": None}

    if not game_start_time_str:
        logger.warning(f"No game_start_time for condition {condition_id}")
        return empty_prices

    try:
        # Parse CLOB game start time
        game_dt = datetime.fromisoformat(game_start_time_str.replace("Z", "+00:00"))
        game_start_ts = int(game_dt.timestamp())

        # Fetch history from 7 days before to 1 hour after game start
        end_ts = game_start_ts + 3600  # 1 hour buffer
        start_ts = game_start_ts - (7 * 24 * 60 * 60)  # 7 days before
    except (ValueError, AttributeError) as e:
        logger.error(f"Failed to parse game_start_time '{game_start_time_str}' for condition {condition_id}: {e}")
        return empty_prices

    # Fetch price histories in parallel with retry logic
    history_1, history_2 = await _fetch_both_histories(session, token_id_1, token_id_2, start_ts, end_ts)

    # Debug logging for empty histories
    if not history_1 or not history_2:
        logger.warning(
            f"Price fetch failed for condition {condition_id}: "
            f"token_1={token_id_1[:8]}... (len={len(history_1)}), "
            f"token_2={token_id_2[:8]}... (len={len(history_2)})"
        )

    # Sort histories for price extraction (filter items with missing timestamp)
    sorted_1 = sorted((h for h in history_1 if "t" in h), key=lambda x: x["t"]) if history_1 else []
    sorted_2 = sorted((h for h in history_2 if "t" in h), key=lambda x: x["t"]) if history_2 else []

    return {
        "p1_close": get_price_at_match_start(sorted_1, game_start_ts),
        "p2_close": get_price_at_match_start(sorted_2, game_start_ts),
    }


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


def find_moneyline_market(markets, event_title):
    """Find the moneyline market from a list of markets.

    For multi-market events, searches through all markets to find the
    full-game moneyline (excludes spreads, totals, half/quarter lines).

    Returns the moneyline market dict, or None if not found.
    """
    # Patterns that indicate non-moneyline markets
    exclude_patterns = [
        "spread", "o/u", "1h", "2h", "1q", "2q", "3q", "4q",
        "over", "under", "total", "points", "margin", "half", "quarter"
    ]

    for market in markets:
        # Parse outcomes
        outcomes_raw = market.get("outcomes", "[]")
        try:
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        except json.JSONDecodeError:
            continue

        # Must have exactly 2 outcomes
        if len(outcomes) != 2:
            continue

        # Skip Yes/No and Over/Under markets
        if "Yes" in outcomes or "No" in outcomes:
            continue
        if "Over" in outcomes or "Under" in outcomes:
            continue

        # Get market question/title
        question = market.get("question", market.get("groupItemTitle", "")).lower()

        # Skip markets with exclude patterns (spreads, totals, halves, etc.)
        if any(pattern in question for pattern in exclude_patterns):
            continue

        # This looks like a moneyline market
        return market

    return None


async def process_event(session, event, tag_id, clob_markets):
    """Process a single event: validate as moneyline, enrich with CLOB data, and flatten.

    Returns flattened dict if valid moneyline event, None otherwise.
    Now enriches gamma-api events with reliable CLOB market data.
    """
    markets = event.get("markets", [])

    if not markets:
        return None

    # Find the moneyline market (handles both single and multi-market events)
    if len(markets) == 1:
        market = markets[0]
    else:
        market = find_moneyline_market(markets, event.get("title", ""))
        if market is None:
            return None

    # Parse JSON once and cache
    outcomes_raw = market.get("outcomes", "[]")
    prices_raw = market.get("outcomePrices", "[]")

    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    except json.JSONDecodeError:
        return None

    # Validate moneyline: exactly 2 outcomes, not Yes/No
    if len(outcomes) != 2:
        return None
    if "Yes" in outcomes or "No" in outcomes:
        return None

    # Extract gamma-api values
    outcome_1 = outcomes[0]
    outcome_2 = outcomes[1]
    price_1 = to_float(prices[0]) if len(prices) > 0 else None
    price_2 = to_float(prices[1]) if len(prices) > 1 else None
    is_closed = event.get("closed", False)

    # === CLOB ENRICHMENT ===
    condition_id = market.get("conditionId", "")
    clob_data = clob_markets.get(condition_id)

    if not clob_data:
        logger.warning(f"No CLOB data for condition_id {condition_id} (title: {event.get('title', '')[:50]})")
        return None  # Skip events without CLOB data

    # Use CLOB token IDs (reliable) instead of gamma clobTokenIds
    token_id_1 = clob_data["token_id_1"]
    token_id_2 = clob_data["token_id_2"]

    # Validate token IDs are not empty
    if not token_id_1 or not token_id_2:
        logger.error(f"Empty token IDs for condition {condition_id} - skipping")
        return None

    # Use CLOB game_start_time instead of gamma endDate
    game_start_time = clob_data["game_start_time"]

    # Use end_date_iso as fallback if game_start_time is missing
    if not game_start_time:
        game_start_time = clob_data.get("end_date_iso")
        if game_start_time:
            logger.debug(f"Using end_date_iso fallback for condition {condition_id}")
        else:
            logger.warning(f"Missing game_start_time and end_date_iso for condition {condition_id}")

    # Use CLOB winner if available, otherwise fall back to gamma price inference
    if clob_data["winner_index"] is not None:
        winner = clob_data["outcome_1"] if clob_data["winner_index"] == 0 else clob_data["outcome_2"]
    else:
        winner = determine_winner(outcomes, prices, is_closed)  # Fallback

    # Warn if CLOB outcomes don't match gamma outcomes (for validation)
    if clob_data["outcome_1"] != outcome_1 or clob_data["outcome_2"] != outcome_2:
        logger.debug(
            f"Outcome mismatch for condition {condition_id}: "
            f"gamma=({outcome_1}, {outcome_2}) vs CLOB=({clob_data['outcome_1']}, {clob_data['outcome_2']})"
        )
        # Use CLOB outcomes as source of truth
        outcome_1 = clob_data["outcome_1"]
        outcome_2 = clob_data["outcome_2"]

    # === END CLOB ENRICHMENT ===

    # Get closing prices using RELIABLE CLOB token IDs and game_start_time
    start_date = event.get("startDate", "")  # Market creation date
    closing = await get_closing_prices(session, token_id_1, token_id_2, condition_id, game_start_time)

    # Determine category and sport
    category, sport = determine_sport_for_event(event, tag_id)

    # Return in chronological order with NEW schema
    return {
        # Category
        "category": category,
        "sport": sport,
        # Identification
        "event_id": event.get("id", ""),
        "condition_id": condition_id,
        "title": event.get("title", ""),
        "slug": event.get("slug", ""),
        # Timing - NEW NAMES
        "market_created_date": format_date_for_excel(start_date),
        "game_start_time": format_date_for_excel(game_start_time),
        # Players
        "outcome_1": outcome_1,
        "outcome_2": outcome_2,
        # Token IDs - NEW
        "token_id_1": token_id_1,
        "token_id_2": token_id_2,
        # Closing prices
        "p1_close": closing.get("p1_close"),
        "p2_close": closing.get("p2_close"),
        # Draw detection - NEW
        "is_50_50_outcome": 1 if clob_data.get("is_50_50_outcome", False) else 0,
        # Result
        "closed": 1 if is_closed else 0,
        "winner": winner,
        # Trading
        "volume": to_float(event.get("volume", 0)),
    }


async def process_events_async(events, tag_id, clob_markets):
    """Process all events concurrently with semaphore for rate limiting.

    Now requires clob_markets dict for enriching events with reliable CLOB data.
    """
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    results = []

    async with aiohttp.ClientSession() as session:
        async def process_with_semaphore(event):
            async with semaphore:
                return await process_event(session, event, tag_id, clob_markets)

        # Create tasks for all events
        tasks = [process_with_semaphore(event) for event in events]

        # Process with progress bar
        with tqdm(total=len(tasks), desc="Processing events", leave=False) as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    results.append(result)
                pbar.update(1)

    return results


def save_to_csv(flattened_events, filename):
    """Save pre-processed moneyline events to a CSV file."""
    if not flattened_events:
        print("No events to save.")
        return

    # NEW SCHEMA with token IDs and renamed date fields
    fieldnames = [
        "category", "sport",                                      # Category
        "event_id", "condition_id", "title", "slug",              # Identification
        "market_created_date", "game_start_time",                 # Timing (RENAMED)
        "outcome_1", "outcome_2",                                 # Players
        "token_id_1", "token_id_2",                               # Token IDs (NEW)
        "p1_close", "p2_close",                                   # Closing prices
        "is_50_50_outcome",                                       # Draw detection (NEW)
        "closed", "winner",                                       # Result
        "volume",                                                 # Trading
    ]

    # Write CSV with proper formatting (QUOTE_MINIMAL avoids quoting numbers)
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(flattened_events)

    print(f"Saved {len(flattened_events)} moneyline events to {filename}")


def main():
    """Main entry point."""
    try:
        # Load sports metadata first for category/sport classification
        print("Loading sports metadata...")
        get_tag_to_sport_map()

        # === NEW: Fetch CLOB markets ONCE at start ===
        print("\nFetching CLOB markets...")
        clob_markets = asyncio.run(fetch_clob_markets_async())
        print(f"Loaded {len(clob_markets)} CLOB markets")
        # === END NEW ===

        all_flattened = []

        for tag_id, sport_codes in SPORTS_TO_FETCH:
            sport_label = "/".join(sport_codes).upper()
            print(f"\nFetching {sport_label} events (tag {tag_id})...")

            events = fetch_all_events(tag_id)
            print(f"Found {len(events)} {sport_label} events from {MIN_YEAR}+")

            if events:
                # Process events for this sport
                print(f"Processing {sport_label} events...")
                # === NEW: Pass clob_markets to processing ===
                flattened = asyncio.run(process_events_async(events, tag_id, clob_markets))
                # === END NEW ===
                all_flattened.extend(flattened)
                print(f"Processed {len(flattened)} {sport_label} moneyline events")

        # Sort by sport first, then by game_start_time (recent to old)
        all_flattened.sort(key=lambda x: (x.get("sport") or "", -(int(datetime.fromisoformat((x.get("game_start_time") or "1970-01-01 00:00:00").replace(" ", "T")).timestamp()) if x.get("game_start_time") else 0)))

        # Write combined CSV
        print(f"\nWriting {len(all_flattened)} total events to {OUTPUT_FILE}...")
        save_to_csv(all_flattened, OUTPUT_FILE)

        # === NEW: DATA QUALITY METRICS ===
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)

        total_events = len(all_flattened)
        closed_events = sum(1 for e in all_flattened if e.get("closed") == 1)

        # Count events with closing prices
        has_prices = sum(
            1 for e in all_flattened
            if e.get("p1_close") is not None and e.get("p2_close") is not None
        )

        # Count closed events with closing prices
        closed_with_prices = sum(
            1 for e in all_flattened
            if e.get("closed") == 1
            and e.get("p1_close") is not None
            and e.get("p2_close") is not None
        )

        # Count events with winners
        has_winner = sum(1 for e in all_flattened if e.get("winner"))

        print(f"Total events:              {total_events:>6}")
        print(f"Closed events:             {closed_events:>6} ({100*closed_events/total_events if total_events > 0 else 0:.1f}%)")
        print(f"Events with prices:        {has_prices:>6} ({100*has_prices/total_events if total_events > 0 else 0:.1f}%)")
        print(f"Closed with prices:        {closed_with_prices:>6} ({100*closed_with_prices/closed_events if closed_events > 0 else 0:.1f}% of closed)")
        print(f"Events with winner:        {has_winner:>6} ({100*has_winner/closed_events if closed_events > 0 else 0:.1f}% of closed)")
        print(f"\nData completeness:         {100*closed_with_prices/closed_events if closed_events > 0 else 0:.1f}% ← TARGET: >90%")
        print("=" * 60)

        # Per-sport breakdown
        print("\nPer-Sport Breakdown:")
        print(f"{'Sport':<20} {'Total':>8} {'Closed':>8} {'With Prices':>12} {'Completeness':>12}")
        print("-" * 65)

        sports = set(e.get("sport") for e in all_flattened)
        for sport in sorted(sports):
            sport_events = [e for e in all_flattened if e.get("sport") == sport]
            sport_total = len(sport_events)
            sport_closed = sum(1 for e in sport_events if e.get("closed") == 1)
            sport_prices = sum(
                1 for e in sport_events
                if e.get("closed") == 1
                and e.get("p1_close") is not None
                and e.get("p2_close") is not None
            )
            completeness = 100 * sport_prices / sport_closed if sport_closed > 0 else 0

            print(f"{sport.upper():<20} {sport_total:>8} {sport_closed:>8} {sport_prices:>12} {completeness:>11.1f}%")

        print("=" * 60)
        # === END DATA QUALITY METRICS ===

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
