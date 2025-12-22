#!/usr/bin/env python3
"""Integration tests for the project."""

import os
import sys
import unittest
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def setUp(self):
        """Set up paths."""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.outputs_dir = self.project_root / 'outputs'

    def test_data_directory_exists(self):
        """Test that data directory exists."""
        self.assertTrue(self.data_dir.exists())

    def test_outputs_directory_exists(self):
        """Test that outputs directory exists."""
        self.assertTrue(self.outputs_dir.exists())

    def test_csv_file_exists(self):
        """Test that fetch_events.csv exists (if data has been fetched)."""
        csv_file = self.data_dir / 'fetch_events.csv'
        if csv_file.exists():
            # If file exists, verify it's valid CSV
            df = pd.read_csv(csv_file)
            
            # Check expected columns
            expected_columns = [
                'sport', 'event_id', 'condition_id', 'title',
                'outcome_1', 'outcome_2', 'p1_close', 'p2_close',
                'winner', 'closed', 'volume', 'token_id_1', 'token_id_2'
            ]
            
            for col in expected_columns:
                self.assertIn(col, df.columns)
            
            # Check that we have data
            self.assertGreater(len(df), 0)
        else:
            # Skip test if data hasn't been fetched yet
            self.skipTest("fetch_events.csv not yet generated")

    def test_visualization_exists(self):
        """Test that visualization exists (if chart has been generated)."""
        chart_file = self.outputs_dir / 'favourite_win_rates.png'
        if chart_file.exists():
            # Verify file is not empty
            self.assertGreater(chart_file.stat().st_size, 0)
        else:
            self.skipTest("favourite_win_rates.png not yet generated")

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists and contains key dependencies."""
        requirements_file = self.project_root / 'requirements.txt'
        self.assertTrue(requirements_file.exists())
        
        with open(requirements_file) as f:
            content = f.read()
            
        # Check for key dependencies
        self.assertIn('pandas', content)
        self.assertIn('matplotlib', content)
        self.assertIn('aiohttp', content)
        self.assertIn('py-clob-client', content)

    def test_readme_exists(self):
        """Test that README.md exists and contains key sections."""
        readme_file = self.project_root / 'README.md'
        self.assertTrue(readme_file.exists())
        
        with open(readme_file) as f:
            content = f.read()
        
        # Check for key sections (may have emojis)
        self.assertIn('# Polymarket Sports Predictability Analysis', content)
        self.assertIn('Key Findings', content)
        self.assertIn('Installation', content)
        self.assertIn('Usage', content)


if __name__ == '__main__':
    unittest.main()
