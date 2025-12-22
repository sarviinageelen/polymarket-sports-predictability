# Tests

This directory contains the test suite for the Polymarket Sports Predictability Analysis project.

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run specific test file
```bash
python -m pytest tests/test_generate_chart.py
```

### Run with coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Run with verbose output
```bash
python -m pytest tests/ -v
```

## Test Structure

- `test_generate_chart.py` - Tests for chart generation and favorite win rate calculations
- `test_fetch_sports.py` - Tests for sports metadata handling
- `test_integration.py` - Integration tests for the full pipeline

## Test Coverage

The test suite covers:
- ✅ Favorite identification logic
- ✅ Win rate calculation
- ✅ Data filtering (closed events only)
- ✅ Missing data handling
- ✅ Sport category merging (NCAAB → CBB)
- ✅ File structure validation
- ✅ CSV schema validation
