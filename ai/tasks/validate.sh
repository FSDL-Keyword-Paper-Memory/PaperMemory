#!/bin/sh

PYTHONPATH=.
pdm run python run/validate.py --path "$1"
