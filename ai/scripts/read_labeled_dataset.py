import os
import re

import pandas as pd

from ai.config import ROOT_DIR

DATA_FOLDER = os.path.join(ROOT_DIR, "ai", 'data', 'Nguyen2007')
KEYS_FOLDER = os.path.join(DATA_FOLDER, 'keys')
PAPERS_FOLDER = os.path.join(DATA_FOLDER, 'docsutf8')

papers_paths = sorted([os.path.join(PAPERS_FOLDER, elem) for elem in os.listdir(PAPERS_FOLDER)], key=lambda x:float(re.findall("(\d+)",os.path.basename(x))[0]))
keys_paths = sorted([os.path.join(KEYS_FOLDER, elem) for elem in os.listdir(KEYS_FOLDER)], key=lambda x:float(re.findall("(\d+)",os.path.basename(x))[0]))

for i, (paper_path, keys_path) in enumerate(zip(papers_paths, keys_paths)):
    key = pd.read_csv(keys_path)
    with open(paper_path) as f:
        lines = f.readlines()

    indices = []
    for i, line in enumerate(lines):
        if 'abstract'.casefold() in line.casefold():
            indices.append(i)
        if 'introduction'.casefold() in line.casefold():
            indices.append(i)
            break

    abstract = " ".join(lines[indices[0]+1: indices[-1]])



