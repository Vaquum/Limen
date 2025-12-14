#!/usr/bin/env python3
"""
Debug window creation issue
"""
import loop
from loop.tests.utils.get_data import get_klines_data
import polars as pl
import numpy as np
from datetime import datetime, timedelta

# Load full dataset
print("Loading full dataset...")
data = get_klines_data()
print(f"Dataset size: {len(data)} rows")
print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
print(f"Columns: {data.columns}")
print(f"Data types: {data.dtypes}")

# Check datetime column specifically
print(f"\nDatetime column analysis:")
print(f"Type: {type(data['datetime'][0])}")
print(f"Sample values: {data['datetime'].head(5).to_list()}")

# Test window creation logic step by step
print(f"\n=== Testing window creation logic ===")

# Sort by datetime
data_sorted = data.sort('datetime')
print(f"Data sorted: {len(data_sorted)} rows")

# Check if datetime conversion is needed
print(f"Schema datetime type: {data_sorted.schema['datetime']}")

if data_sorted.schema['datetime'] != pl.Datetime:
    print("Converting datetime column...")
    try:
        data_sorted = data_sorted.with_columns(
            pl.col('datetime').str.strptime(pl.Datetime)
        )
        print("Datetime conversion successful")
    except Exception as e:
        print(f"Datetime conversion failed: {e}")

# Get date range after conversion
start_date = data_sorted['datetime'].min()
end_date = data_sorted['datetime'].max()
print(f"Start date: {start_date} (type: {type(start_date)})")
print(f"End date: {end_date} (type: {type(end_date)})")

# Handle datetime difference calculation properly
if hasattr(start_date, 'total_seconds'):
    # Standard datetime objects
    total_duration = (end_date - start_date).total_seconds() / 86400  # days
else:
    # Polars datetime - convert to python datetime first
    start_py = start_date.to_py() if hasattr(start_date, 'to_py') else start_date
    end_py = end_date.to_py() if hasattr(end_date, 'to_py') else end_date
    total_duration = (end_py - start_py).total_seconds() / 86400  # days

print(f"Total duration: {total_duration} days")

# Test window creation for 1-day windows
window_days = 1
print(f"\n=== Testing {window_days}-day window creation ===")

if total_duration <= window_days:
    print(f"Total data ({total_duration:.2f} days) <= window size ({window_days} days)")
    print("Would create 1 window with all data")
else:
    print(f"Total data ({total_duration:.2f} days) > window size ({window_days} days)")
    print("Proceeding with window creation...")

    # Test creating first few windows
    stride_days = 1  # Daily stride
    current_start = start_date
    window_count = 0
    max_test_windows = 5

    print(f"Starting window creation with stride_days={stride_days}")

    while current_start < end_date and window_count < max_test_windows:
        window_end = current_start + timedelta(days=window_days)

        if window_end > end_date:
            window_end = end_date

        print(f"Window {window_count + 1}: {current_start} to {window_end}")

        # Filter data for this window
        try:
            window_data = data_sorted.filter(
                (pl.col('datetime') >= current_start) &
                (pl.col('datetime') <= window_end)
            )
            print(f"  Window data size: {len(window_data)} rows")

            # Check if meets minimum requirement
            if len(window_data) >= 50:
                print(f"  ✓ Window {window_count + 1} has sufficient data")
                window_count += 1
            else:
                print(f"  ✗ Window {window_count + 1} insufficient data ({len(window_data)} < 50)")

        except Exception as e:
            print(f"  Error creating window: {e}")

        # Move to next window
        current_start += timedelta(days=stride_days)

    print(f"Created {window_count} valid windows out of {max_test_windows} tested")