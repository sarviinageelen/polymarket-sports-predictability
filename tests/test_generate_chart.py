#!/usr/bin/env python3
"""Tests for generate_chart.py"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generate_chart import (
    calculate_favourite_win_rates,
    SPORT_DISPLAY_NAMES
)


class TestGenerateChart(unittest.TestCase):
    """Test suite for chart generation functions."""

    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'sport': ['nba', 'nba', 'mlb', 'mlb', 'atp', 'atp'],
            'closed': [1, 1, 1, 1, 1, 1],
            'p1_close': [0.7, 0.6, 0.55, 0.48, 0.8, 0.45],
            'p2_close': [0.3, 0.4, 0.45, 0.52, 0.2, 0.55],
            'outcome_1': ['Lakers', 'Celtics', 'Yankees', 'Dodgers', 'Nadal', 'Federer'],
            'outcome_2': ['Celtics', 'Bucks', 'Red Sox', 'Giants', 'Djokovic', 'Murray'],
            'winner': ['Lakers', 'Celtics', 'Red Sox', 'Giants', 'Nadal', 'Murray']
        })

    def test_calculate_favourite_win_rates_basic(self):
        """Test basic favorite win rate calculation."""
        results = calculate_favourite_win_rates(self.sample_data)

        # Check that we get results for all sports
        self.assertIn('nba', results)
        self.assertIn('mlb', results)
        self.assertIn('atp', results)

        # NBA: 2 events, Lakers (0.7) beats Celtics (0.3), Celtics (0.6) beats Bucks (0.4)
        # Both favorites won, so win rate should be 100%
        nba_rate, nba_wins, nba_total = results['nba']
        self.assertEqual(nba_total, 2)
        self.assertEqual(nba_wins, 2)
        self.assertEqual(nba_rate, 100.0)

        # ATP: 2 events, Nadal (0.8) beats Djokovic (0.2), Murray (0.55) beats Federer (0.45)
        # Both favorites won
        atp_rate, atp_wins, atp_total = results['atp']
        self.assertEqual(atp_total, 2)
        self.assertEqual(atp_wins, 2)

    def test_calculate_favourite_win_rates_skips_ties(self):
        """Test that equal prices are skipped."""
        tie_data = pd.DataFrame({
            'sport': ['nba'],
            'closed': [1],
            'p1_close': [0.5],
            'p2_close': [0.5],
            'outcome_1': ['Lakers'],
            'outcome_2': ['Celtics'],
            'winner': ['Lakers']
        })
        
        results = calculate_favourite_win_rates(tie_data)
        
        # Should have no results because prices are equal (no clear favorite)
        self.assertNotIn('nba', results)

    def test_calculate_favourite_win_rates_filters_closed_only(self):
        """Test that only closed events are analyzed."""
        mixed_data = self.sample_data.copy()
        mixed_data.loc[0, 'closed'] = 0  # Make first event not closed
        
        results = calculate_favourite_win_rates(mixed_data)
        
        # NBA should now have only 1 event instead of 2
        nba_rate, nba_wins, nba_total = results['nba']
        self.assertEqual(nba_total, 1)

    def test_calculate_favourite_win_rates_handles_missing_data(self):
        """Test that events with missing data are skipped."""
        missing_data = self.sample_data.copy()
        missing_data.loc[0, 'p1_close'] = None  # Set to missing
        
        results = calculate_favourite_win_rates(missing_data)
        
        # NBA should have only 1 valid event
        nba_rate, nba_wins, nba_total = results['nba']
        self.assertEqual(nba_total, 1)

    def test_sport_display_names(self):
        """Test that all expected sports have display names."""
        expected_sports = ['atp', 'wta', 'nba', 'nfl', 'mlb', 'cfb', 'ncaab', 'cbb']
        
        for sport in expected_sports:
            self.assertIn(sport, SPORT_DISPLAY_NAMES)
            self.assertTrue(len(SPORT_DISPLAY_NAMES[sport]) > 0)

    def test_calculate_favourite_win_rates_merges_ncaab_cbb(self):
        """Test that ncaab is merged into cbb category."""
        ncaab_data = pd.DataFrame({
            'sport': ['ncaab', 'cbb'],
            'closed': [1, 1],
            'p1_close': [0.7, 0.6],
            'p2_close': [0.3, 0.4],
            'outcome_1': ['Duke', 'UNC'],
            'outcome_2': ['UNC', 'Duke'],
            'winner': ['Duke', 'UNC']
        })
        
        results = calculate_favourite_win_rates(ncaab_data)
        
        # Should be merged into 'cbb' category
        self.assertIn('cbb', results)
        self.assertNotIn('ncaab', results)
        
        # Should have 2 total events
        cbb_rate, cbb_wins, cbb_total = results['cbb']
        self.assertEqual(cbb_total, 2)


if __name__ == '__main__':
    unittest.main()
