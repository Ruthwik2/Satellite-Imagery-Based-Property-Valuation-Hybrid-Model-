#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
python src/hybrid_property_valuation.py
