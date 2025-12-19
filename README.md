# pm-tennis

Python scripts to fetch sports and tennis event data from the Polymarket API.

## Features

- **Sports Metadata Fetcher** - Fetches all sports from Polymarket with automatic category classification (soccer, tennis, basketball, etc.)
- **Tennis Events Fetcher** - Fetches tennis moneyline (head-to-head) events with match outcomes and winner data

## Requirements

- Python 3.7+
- `requests` library

```bash
pip install requests
```

## Usage

### Fetch Sports Metadata

```bash
python fetch_polymarket_sports.py
```

**Output:** `polymarket_sports.csv`

| Column | Description |
|--------|-------------|
| category | Sport category (soccer, tennis, basketball, etc.) |
| sport | Sport code (atp, wta, nfl, etc.) |
| image | Logo URL |
| resolution | Official league website |
| ordering | Display order preference |
| tags | Category tag IDs |
| series | Series identifier |

### Fetch Tennis Events

```bash
python fetch_tennis_events.py
```

**Output:** `polymarket_tennis_events.csv`

| Column | Description |
|--------|-------------|
| event_id | Unique event identifier |
| title | Match title (e.g., "US Open: Zverev vs. Nakashima") |
| slug | URL-friendly identifier |
| start_date | Event start date |
| end_date | Event end date |
| active | Is event active |
| closed | Is event closed/resolved |
| liquidity | Total liquidity |
| volume | Total volume traded |
| outcome_1 | First player |
| outcome_2 | Second player |
| price_1 | Price for outcome_1 |
| price_2 | Price for outcome_2 |
| winner | Winning player (for closed events) |

## API Reference

- [Polymarket Sports API](https://docs.polymarket.com/api-reference/sports/get-sports-metadata-information)
- [Polymarket Events API](https://docs.polymarket.com/api-reference/events/get-event-by-id)

## License

MIT
