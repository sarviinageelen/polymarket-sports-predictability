# Polymarket Sports Predictability Analysis

> Analyzing 7,141 sports betting markets to determine how often favorites actually win

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Key Findings

![Favorite Win Rates by Sport](outputs/favourite_win_rates.png)

**TL;DR:** Favorites win at vastly different rates across sports:
- 🏀 **College Basketball (CBB):** 81% favorite win rate (n=48)
- 🏈 **College Football (CFB):** 75% (n=793)
- 🎾 **ATP Tennis:** 69% (n=1,748)
- 🏀 **NBA:** 68% (n=1,393)
- 🎾 **WTA Tennis:** 67% (n=18)
- 🏈 **NFL:** 65% (n=294)
- ⚾ **MLB:** 56% (n=2,397) - Most unpredictable!

**Data Quality:** 7,141 total events with **99.7% completeness** - achieved through hybrid API architecture.

---

## 🎯 Project Overview

This project analyzes Polymarket prediction markets to answer: **"How often do favorites win in different sports?"**

### Why This Matters
- **Sports analytics:** Understanding predictability across different sports
- **Market efficiency:** Testing if betting markets accurately reflect win probabilities
- **Technical showcase:** Demonstrates API integration, data engineering, and business logic skills

### The Challenge
Polymarket's public Gamma API had **89% data loss** for pricing information. This project solved it by:
1. Building a **hybrid API architecture** (Gamma API + CLOB API)
2. Implementing **token ID matching** for reliable data enrichment
3. Achieving **99.7% data completeness** (from 8.2%)

---

## 🛠️ Technical Approach

### Architecture

```
┌─────────────────┐
│  Gamma API      │  ← Fetch sports metadata & events by tag
│  (Sport Tags)   │
└────────┬────────┘
         │
         ├──→ 7,141 events
         │
         ▼
┌─────────────────┐
│  CLOB API       │  ← Enrich with pricing & settlement data
│  (Token IDs)    │
└────────┬────────┘
         │
         ├──→ 99.7% completeness
         │
         ▼
┌─────────────────┐
│  Analysis       │  ← Calculate favorite win rates
│  (Pandas)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualization  │  ← Generate charts
│  (Matplotlib)   │
└─────────────────┘
```

### Key Technical Features

**1. Hybrid API Integration**
- **Gamma API:** Sport-based event filtering via tags
- **CLOB API:** Reliable pricing data via token IDs
- **Async/Await:** Concurrent requests with \`aiohttp\`

**2. Data Quality Engineering**
- Retry logic with exponential backoff
- Rate limiting via semaphores
- Comprehensive error handling
- Data validation & cleaning

**3. Business Logic**
- Favorite identification: \`max(p1_close, p2_close)\`
- Win rate calculation: \`(favorite_wins / total_closed) * 100%\`
- Sport-by-sport aggregation
- Statistical significance tracking

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

\`\`\`bash
# Clone repository
git clone https://github.com/yourusername/polymarket-sports-predictability.git
cd polymarket-sports-predictability

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Dependencies
- \`pandas\` - Data manipulation
- \`matplotlib\` - Visualization
- \`aiohttp\` - Async HTTP requests
- \`py-clob-client\` - Polymarket CLOB API SDK
- \`requests\` - HTTP requests
- \`tqdm\` - Progress bars

---

## 🚀 Usage

### 1. Fetch Sports Data
\`\`\`bash
python src/fetch_sports.py
\`\`\`
**Output:** \`data/fetch_sports.csv\` (sports metadata)

### 2. Fetch Event Data
\`\`\`bash
python src/fetch_events.py
\`\`\`
**Output:** \`data/fetch_events.csv\` (7,141 events with pricing)
**Duration:** ~60-90 minutes (API rate limits)

### 3. Generate Visualization
\`\`\`bash
python src/generate_chart.py
\`\`\`
**Output:** \`outputs/favourite_win_rates.png\`

---

## 📁 Project Structure

\`\`\`
polymarket-sports-predictability/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore                  # Git ignore rules
├── GITHUB_SETUP.md             # GitHub repository setup guide
│
├── src/                        # Source code
│   ├── fetch_sports.py         # Fetch sports metadata from Gamma API
│   ├── fetch_events.py         # Fetch & enrich event data (hybrid API)
│   └── generate_chart.py       # Analyze data & generate visualizations
│
├── tests/                      # Test suite
│   ├── test_fetch_sports.py    # Tests for sports metadata
│   ├── test_generate_chart.py  # Tests for chart generation
│   └── test_integration.py     # Integration tests
│
├── data/                       # Generated data
│   ├── fetch_events.csv        # Event data with pricing
│   └── fetch_sports.csv        # Sports metadata
│
└── outputs/                    # Generated visualizations
    └── favourite_win_rates.png # Main chart
\`\`\`

---

## 📊 Data Schema

### \`data/fetch_events.csv\`

| Column | Type | Description |
|--------|------|-------------|
| \`sport\` | str | Sport code (atp, wta, nba, nfl, mlb, cfb, cbb) |
| \`event_id\` | int | Gamma API event ID |
| \`condition_id\` | str | CLOB API condition ID (for matching) |
| \`title\` | str | Event title (e.g., "Lakers vs Celtics") |
| \`outcome_1\` | str | First outcome (e.g., "Lakers") |
| \`outcome_2\` | str | Second outcome (e.g., "Celtics") |
| \`p1_close\` | float | Outcome 1 closing price (0-1) |
| \`p2_close\` | float | Outcome 2 closing price (0-1) |
| \`winner\` | str | Winning outcome (if closed) |
| \`closed\` | int | 1 if market closed, 0 if active |
| \`volume\` | float | Trading volume ($) |
| \`token_id_1\` | str | CLOB token ID for outcome 1 |
| \`token_id_2\` | str | CLOB token ID for outcome 2 |

---

## 🔍 Technical Deep Dive

### API Integration Challenge

**Problem:** Gamma API returns events but has 89% missing data for prices.

**Solution:** Hybrid architecture
1. Use Gamma API to fetch events by sport tags
2. Extract \`condition_id\` from events
3. Use CLOB API to fetch pricing data via \`condition_id\`
4. Match token IDs to get reliable pricing

**Code Example:**
\`\`\`python
# Gamma API - Get events
gamma_response = requests.get(
    f"https://gamma-api.polymarket.com/events",
    params={"tag": sport_tag}
)

# CLOB API - Get pricing
clob_markets = await fetch_clob_markets_async()
for event in events:
    condition_id = event["condition_id"]
    # Match in CLOB data for pricing
    clob_data = clob_markets.get(condition_id)
\`\`\`

### Data Quality Improvement

**Before (Gamma API only):**
- 7,141 total events
- 588 events with prices (8.2%)
- ❌ Insufficient for statistical analysis

**After (Hybrid Gamma + CLOB):**
- 7,141 total events
- 7,122 events with prices (99.7%)
- ✅ Statistically significant across all sports

### Async Processing

Used \`aiohttp\` with semaphores for concurrent API requests while respecting rate limits:

\`\`\`python
async with aiohttp.ClientSession() as session:
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
    tasks = [fetch_with_semaphore(semaphore, session, url)
             for url in urls]
    results = await asyncio.gather(*tasks)
\`\`\`

---

## 📈 Analysis Methodology

### Favorite Identification
The favorite is defined as the outcome with the higher closing price on Polymarket:

\`\`\`python
favorite_prob = max(p1_close, p2_close)
favorite = outcome_1 if p1_close > p2_close else outcome_2
\`\`\`

### Win Rate Calculation
Win rate measures how often the favorite actually won:

\`\`\`python
favorite_wins = (favorite == winner).sum()
win_rate = (favorite_wins / total_games) * 100
\`\`\`

### Sports Covered
- **Tennis:** ATP (1,748 events), WTA (18 events)
- **Basketball:** NBA (1,393), College Basketball (48)
- **Football:** NFL (294), College Football (793)
- **Baseball:** MLB (2,397)

---

## 🎨 Visualization

### Chart Features
- **16:9 widescreen format** (1920×1080 optimized)
- **The Athletic-inspired styling** (professional sports media aesthetic)
- **Sorted by win rate** (highest to lowest)
- **Sample sizes included** (statistical transparency)
- **Accessible color scheme** (cream background, teal bars)

### Design Choices
- Horizontal bars for easy sport comparison
- Large, bold percentages for readability
- Clean, minimal design (no chart junk)
- Professional typography

---

## 💡 Key Insights

### 1. MLB is the Least Predictable Sport
With only 56% favorite win rate, Major League Baseball shows the highest degree of unpredictability. This aligns with the sport's high variance - even great teams lose 40% of their games in a season.

### 2. College Basketball Favorites Dominate
At 81%, CBB favorites win more consistently than any other sport analyzed. This could reflect:
- Larger talent gaps between teams
- Home court advantage in college sports
- Tournament selection bias (better teams get favorable matchups)

### 3. Tennis and Basketball Show Similar Patterns
ATP (69%), NBA (68%), and WTA (67%) cluster together, suggesting similar levels of competitive balance across these sports.

### 4. Football Falls in the Middle
NFL (65%) and CFB (75%) show moderate predictability, with college football being more predictable - likely due to larger talent disparities between top and bottom teams.

### 5. Market Efficiency
The varying win rates across sports suggest Polymarket users are reasonably calibrated in their predictions. Markets aren't showing extreme over-confidence (>90% favorites winning <50%) or under-confidence (50% favorites winning >70%).

---

## 🔮 Future Improvements

- [ ] **Time-series analysis** - How do win rates change over seasons?
- [ ] **Market calibration curves** - Are 70% favorites actually winning 70%?
- [ ] **Volume vs accuracy** - Do higher-volume markets predict better?
- [ ] **Upset analysis** - When do underdogs win and why?
- [ ] **Live dashboard** - Real-time market tracking
- [ ] **More sports** - Soccer, hockey, esports, MMA

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Polymarket** for providing public APIs
- **py-clob-client** SDK by Polymarket
- **The Athletic** for design inspiration
- **Python ecosystem** (pandas, matplotlib, aiohttp)

---

## 📧 Contact

Questions? Open an issue or reach out via email.

---

*Built with Python 🐍 | Powered by Polymarket API 📊 | Data-driven insights ✨*
