#!/bin/sh

PYTHONPATH=.
pdm run python run/predict.py --model "$1" --threshold "$2" --abstract "$3"
