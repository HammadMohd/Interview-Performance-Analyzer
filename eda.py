#!/usr/bin/env python3
"""
Root entry point for running Exploratory Data Analysis on merged features.
Usage:
    python eda.py
    python eda.py --csv data/features/merged_features.csv --output-dir reports/eda
"""

import sys
from src.eda_merged_features import main

if __name__ == '__main__':
    main()
