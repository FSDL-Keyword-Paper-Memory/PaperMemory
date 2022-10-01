#!/bin/sh

PYTHONPATH=.
pdm run python scripts/read_labeled_dataset.py --path $1
