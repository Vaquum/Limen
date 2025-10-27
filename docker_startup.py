#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pandas as pd
from pathlib import Path

def create_sample_data():
    """Create sample data from existing CSV files for Streamlit app"""

    # Check if we have CSV data files
    datasets_dir = Path('datasets')
    if not datasets_dir.exists():
        print("No datasets directory found")
        return False

    csv_files = list(datasets_dir.glob('*.csv'))
    if not csv_files:
        print("No CSV files found in datasets directory")
        return False

    # Use the largest CSV file as sample data
    largest_file = max(csv_files, key=lambda f: f.stat().st_size)
    print(f"Using {largest_file} as sample data source")

    try:
        # Read the CSV data
        df = pd.read_csv(largest_file)

        # Create sample parquet file for Streamlit app
        output_path = '/tmp/historical_data.parquet'
        df.to_parquet(output_path)

        print(f"Created sample data at {output_path}")
        print(f"Data shape: {df.shape}")

        return True

    except Exception as e:
        print(f"Error creating sample data: {e}")
        return False

def run_streamlit_with_data():
    """Run Streamlit app with the sample data"""
    import subprocess

    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        'loop/explorer/streamlit_app.py',
        '--server.address', '0.0.0.0',
        '--server.port', '8501',
        '--server.headless', 'true',
        '--',
        '--data', '/tmp/historical_data.parquet'
    ]

    print("Starting Streamlit app...")
    print(f"Command: {' '.join(cmd)}")

    # Set environment variable as fallback
    os.environ['DATA_PARQUET'] = '/tmp/historical_data.parquet'

    subprocess.run(cmd)

if __name__ == '__main__':
    print("Docker startup: Creating sample data...")

    if create_sample_data():
        print("Docker startup: Sample data created successfully")
        run_streamlit_with_data()
    else:
        print("Docker startup: Failed to create sample data")
        sys.exit(1)