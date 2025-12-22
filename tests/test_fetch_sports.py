#!/usr/bin/env python3
"""Tests for fetch_sports.py"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fetch_sports import SPORT_CATEGORIES


class TestFetchSports(unittest.TestCase):
    """Test suite for sports metadata fetching."""

    def test_sport_categories_coverage(self):
        """Test that major sports are in SPORT_CATEGORIES."""
        expected_sports = {
            'atp': 'tennis',
            'wta': 'tennis',
            'nba': 'basketball',
            'ncaab': 'basketball',
            'nfl': 'american_football',
            'cfb': 'american_football',
            'mlb': 'baseball'
        }
        
        for sport_code, expected_category in expected_sports.items():
            self.assertIn(sport_code, SPORT_CATEGORIES)
            self.assertEqual(SPORT_CATEGORIES[sport_code], expected_category)

    def test_sport_categories_unique_values(self):
        """Test that sport categories are valid."""
        valid_categories = {
            'soccer', 'basketball', 'american_football',
            'baseball', 'ice_hockey', 'cricket', 'tennis', 'mma', 'esports'
        }

        for category in SPORT_CATEGORIES.values():
            self.assertIn(category, valid_categories)

    def test_sport_categories_lowercase(self):
        """Test that all sport codes are lowercase."""
        for sport_code in SPORT_CATEGORIES.keys():
            self.assertEqual(sport_code, sport_code.lower())


if __name__ == '__main__':
    unittest.main()
