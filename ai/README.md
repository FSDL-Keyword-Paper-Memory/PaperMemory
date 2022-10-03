# FSDL Keyword PaperMemory

## Requirements

1. Python 3.10 or above
2. PDM (as a dependency manager) -> download and install from [here](https://pdm.fming.dev/latest/#installation)

## Setup environment

Below code will create a virtual environment as well as download and install required packages.  

```bash
cd ai
pdm sync
```

## Abstract scraper

To prepare dataset that in particular consists of abstracts downloaded from [arxiv.org](https://arxiv.org/) one should run the script with a code from below:

```bash
make get-fresh-data
```

Output will be a JSON file that will be saved to `ai/data` directory, log will be stored in `ai/logs`.

## Abstract cleaner

The abstract cleaner was created based on simple EDA that is shown in `notebooks/simple_eda.ipynb`. To preprocess and clean abstracts downloaded from [arxiv.org](https://arxiv.org/) one should run the scrip with a code from below:

```bash
make clean-abstracts path=<PATH-TO-SCRAPED-ARXIV-DATASET>
```

Output will be a plain text file (abstract per row) that will be saved to `ai/data` directory with `_cleaned.txt` suffix, log will be stored in `ai/logs`.

## Preparing devset

Devset was created from the [Krapivin2009 dataset](https://github.com/LIAAD/KeywordExtractor-Datasets#krapivin2009). It is the biggest dataset in terms of documents, with 2,304 full papers from the Computer Science domain, which were published by ACM in the period ranging from 2003 to 2005. The papers were downloaded from CiteSeerX Autonomous Digital Library and each one has its keywords assigned by the authors and verified by the reviewers. From this dataset only abstracts were used and keywords that were present within corresponding abstracts.  
To prepare devset one should run the scrip with a code from below:

```bash
make prepare-devset path=<PATH-TO-DOWNLOADED-AND-UNPACKED-KRAPIVIN2009-DATASET>
```

## Validate

To validate the model on the devset one should run the scrip with a code from below:

```bash
make validate path=<PATH-TO-DEVSET>
```

There is possibility to try different models and thresholds. Adjust `MODELS` and `THRESHOLDS` variables in `ai/run/validate.py` if needed.

## Predict

To make a prediction one should run the scrip with a code from below:

```bash
make predict model=<MODEL-NAME> threshold=<THRESHOLD> abstract=<ABSTRACT>
```

<u>`<ABSTRACT>` must be contained in `"` (double quote marks)!</u>

## API

To create a docker container with Flask API one should run the scrip with a code from below:

```bash
make api
```

Endpoint:  

* `/predict`
* port: 2137

Example request to get prediction:

```bash
curl -H "Content-Type: application/json" \
     -d '{"abstract": "<ABSTRACT>"}' \
     -X POST <URL>:2137/predict
```
