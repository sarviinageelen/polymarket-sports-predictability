# Polymarket Sports Predictability Analysis

Statistical analysis of favorite win rates across sports prediction markets using hybrid API architecture.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project analyzes sports prediction market efficiency on Polymarket. The primary research question: **What is the empirical win rate of favorites across different professional sports?**

The analysis processes 7,141 closed betting markets across seven sports (ATP, WTA, NBA, NFL, MLB, CFB, CBB), achieving 99.7% data completeness through a hybrid API integration approach.

### Key Results

![Favorite Win Rates by Sport](outputs/favourite_win_rates.png)

| Sport | Favorite Win Rate | Sample Size | Events Analyzed |
|-------|------------------|-------------|-----------------|
| College Basketball | 81.2% | 39/48 | 48 |
| College Football | 74.8% | 593/793 | 793 |
| ATP Tennis | 69.3% | 1,212/1,748 | 1,748 |
| NBA Basketball | 67.8% | 945/1,393 | 1,393 |
| WTA Tennis | 66.7% | 12/18 | 18 |
| NFL Football | 65.3% | 192/294 | 294 |
| MLB Baseball | 56.4% | 1,352/2,397 | 2,397 |

## Architecture

The system implements a multi-stage data pipeline integrating two Polymarket APIs:

```
┌──────────────────┐
│   Gamma API      │  Sport-based event filtering via tag IDs
│  (Event Catalog) │  Fetches: event metadata, participants, market structure
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    CLOB API      │  Token-based pricing enrichment
│  (Order Book)    │  Fetches: closing prices, settlement data, volume
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Analysis Engine  │  Win rate calculation and aggregation
│    (Pandas)      │  Logic: identify favorite → validate winner → compute rates
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Visualization   │  Statistical output generation
│  (Matplotlib)    │  Format: 16:9 horizontal bar charts
└──────────────────┘
```

## Key Features

- **Multi-sport support**: ATP, WTA, NBA, NFL, MLB, CFB, CBB
- **Hybrid API architecture**: Combines Gamma API (events) + CLOB API (pricing)
- **99.7% data completeness**: Token ID matching resolves missing price data
- **Async pipeline**: Concurrent fetching with aiohttp and rate limiting
- **Error tracking**: Comprehensive retry logic with exponential backoff
- **Professional visualization**: 16:9 charts with sample size annotations

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/polymarket-sports-predictability.git
cd polymarket-sports-predictability

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Fetch Sports Metadata

```bash
python src/fetch_sports.py
```

**Output**: `data/fetch_sports.csv` (~5 seconds)

### Step 2: Fetch Event Data

```bash
python src/fetch_events.py
```

**Output**: `data/fetch_events.csv` (60-90 minutes for 7,141 events)

### Step 3: Generate Analysis

```bash
python src/generate_chart.py
```

**Output**: `outputs/favourite_win_rates.png` (~2 seconds)

## API Integration

The pipeline uses two Polymarket APIs:

1. **Gamma API** (`https://gamma-api.polymarket.com/events`) - Event discovery via sport tags
2. **CLOB API** (`https://clob.polymarket.com`) - Reliable pricing and settlement data

The hybrid approach resolves Gamma API's 89% missing price data by matching events via `condition_id` to CLOB market data.

## Output Structure

```
data/
├── fetch_events.csv    # 7,141 events with pricing and settlement
└── fetch_sports.csv    # Sports metadata and tag mappings

outputs/
└── favourite_win_rates.png    # Win rate visualization (1920×1080)
```

## Project Structure

```
polymarket-sports-predictability/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── fetch_sports.py      # Sports metadata fetcher
│   ├── fetch_events.py      # Event data pipeline
│   └── generate_chart.py    # Win rate analysis
├── tests/
│   ├── test_fetch_sports.py
│   ├── test_generate_chart.py
│   └── test_integration.py
├── data/                    # Generated datasets
└── outputs/                 # Generated visualizations
```

## Disclaimer

This project is for educational and research purposes only. The analysis is based on historical market data and should not be construed as investment advice.

- Past performance does not guarantee future results
- Users should comply with all applicable laws and Polymarket's terms of service

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
