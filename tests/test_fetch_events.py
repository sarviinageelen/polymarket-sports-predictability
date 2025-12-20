import unittest
from unittest.mock import patch

import fetch_events as fte


class DummyTqdm:
    def __init__(self, *args, **kwargs):
        self.updated = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n):
        self.updated += n


class TestDateHelpers(unittest.TestCase):
    def test_format_date_for_excel_valid(self):
        value = "2025-03-01T12:34:56Z"
        self.assertEqual(fte.format_date_for_excel(value), "2025-03-01 12:34:56")

    def test_format_date_for_excel_invalid(self):
        self.assertIsNone(fte.format_date_for_excel("not-a-date"))
        self.assertIsNone(fte.format_date_for_excel(None))

    def test_is_2025_or_later(self):
        self.assertTrue(fte.is_2025_or_later("2025-01-01T00:00:00Z"))
        self.assertFalse(fte.is_2025_or_later("2024-12-31T23:59:59Z"))
        self.assertFalse(fte.is_2025_or_later(""))


class TestPriceHelpers(unittest.TestCase):
    def test_derive_closing_from_trades(self):
        trades = [
            {"timestamp": 2, "asset": "token2", "price": 0.4},
            {"timestamp": 1, "asset": "token1", "price": 0.5},
            {"timestamp": 3, "asset": "token1", "price": 1.0},
            {"timestamp": 4, "asset": "token2", "price": 0.0},
            {"timestamp": 5, "asset": "token1", "price": 0.55},
        ]
        closing = fte.derive_closing_from_trades(trades, "token1", "token2")
        self.assertEqual(closing, {"p1_close": 0.55, "p2_close": 0.4})

    def test_get_true_closing_price(self):
        history = [
            {"t": 1, "p": 0.5},
            {"t": 2, "p": 1.0},
            {"t": 3, "p": 0.02},
        ]
        self.assertEqual(fte.get_true_closing_price(history), 0.5)
        self.assertIsNone(fte.get_true_closing_price([]))

    def test_determine_winner(self):
        outcomes = ["Player A", "Player B"]
        prices = ["1.0", "0.0"]
        self.assertEqual(fte.determine_winner(outcomes, prices, True), "Player A")
        self.assertIsNone(fte.determine_winner(outcomes, prices, False))

    def test_to_float(self):
        self.assertEqual(fte.to_float("3.25"), 3.25)
        self.assertIsNone(fte.to_float("not-a-number"))
        self.assertIsNone(fte.to_float(None))
        self.assertIsNone(fte.to_float(""))


class TestFetchAndProcess(unittest.TestCase):
    def test_fetch_all_tennis_events_filters_by_year(self):
        events_page_1 = [
            {"id": 1, "startDate": "2025-01-01T00:00:00Z"},
            {"id": 2, "startDate": "2024-12-31T00:00:00Z"},
        ]
        with patch("fetch_events.fetch_tennis_events_page", side_effect=[events_page_1, []]):
            with patch("fetch_events.tqdm", DummyTqdm):
                events = fte.fetch_all_tennis_events()
        self.assertEqual([event["id"] for event in events], [1])

    def test_get_closing_prices_falls_back_to_trades(self):
        trades = [
            {"timestamp": 1, "asset": "token1", "price": 0.45},
            {"timestamp": 2, "asset": "token2", "price": 0.55},
        ]
        with patch("fetch_events._fetch_both_histories", return_value=([], [])):
            with patch("fetch_events.fetch_trades_for_condition", return_value=trades):
                closing = fte.get_closing_prices(
                    "token1",
                    "token2",
                    "condition123",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T12:00:00Z",
                )
        self.assertEqual(closing, {"p1_close": 0.45, "p2_close": 0.55})

    def test_process_event_valid(self):
        event = {
            "id": "event123",
            "title": "Player A vs Player B",
            "slug": "player-a-vs-player-b",
            "startDate": "2025-03-01T12:34:56Z",
            "endDate": "2025-03-01T14:00:00Z",
            "volume": "123.45",
            "closed": True,
            "markets": [
                {
                    "outcomes": "[\"Player A\", \"Player B\"]",
                    "outcomePrices": "[1.0, 0.0]",
                    "clobTokenIds": "[\"token1\", \"token2\"]",
                    "conditionId": "condition123",
                }
            ],
        }
        with patch("fetch_events.get_closing_prices", return_value={"p1_close": 0.6, "p2_close": 0.4}) as mock_closing:
            with patch("fetch_events.determine_sport_for_event", return_value=("tennis", "atp")):
                result = fte.process_event(event)

        mock_closing.assert_called_once_with(
            "token1",
            "token2",
            "condition123",
            "2025-03-01T12:34:56Z",
            "2025-03-01T14:00:00Z",
        )
        self.assertEqual(result["category"], "tennis")
        self.assertEqual(result["sport"], "atp")
        self.assertEqual(result["event_id"], "event123")
        self.assertEqual(result["outcome_1"], "Player A")
        self.assertEqual(result["outcome_2"], "Player B")
        self.assertEqual(result["p1_close"], 0.6)
        self.assertEqual(result["p2_close"], 0.4)
        self.assertEqual(result["winner"], "Player A")
        self.assertEqual(result["start_date"], "2025-03-01 12:34:56")
        self.assertEqual(result["end_date"], "2025-03-01 14:00:00")
        self.assertEqual(result["volume"], 123.45)

    def test_process_event_rejects_yes_no(self):
        event = {
            "markets": [
                {
                    "outcomes": "[\"Yes\", \"No\"]",
                    "outcomePrices": "[0.6, 0.4]",
                    "clobTokenIds": "[\"token1\", \"token2\"]",
                }
            ]
        }
        self.assertIsNone(fte.process_event(event))

    def test_process_event_rejects_invalid_json(self):
        event = {
            "markets": [
                {
                    "outcomes": "[\"Player A\", \"Player B\"",
                    "outcomePrices": "[0.6, 0.4]",
                    "clobTokenIds": "[\"token1\", \"token2\"]",
                }
            ]
        }
        self.assertIsNone(fte.process_event(event))


if __name__ == "__main__":
    unittest.main()
