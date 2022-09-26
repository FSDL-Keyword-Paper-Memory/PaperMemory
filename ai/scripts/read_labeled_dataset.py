import os
import re

import pandas as pd

from ai.config import ROOT_DIR

DATA_FOLDER = os.path.join(ROOT_DIR, "ai", "data", "Krapivin2009")
KEYS_FOLDER = os.path.join(DATA_FOLDER, "keys")
PAPERS_FOLDER = os.path.join(DATA_FOLDER, "docsutf8")

papers_paths = sorted(
    [os.path.join(PAPERS_FOLDER, elem) for elem in os.listdir(PAPERS_FOLDER)],
    key=lambda x: float(re.findall("(\d+)", os.path.basename(x))[0]),
)
keys_paths = sorted(
    [os.path.join(KEYS_FOLDER, elem) for elem in os.listdir(KEYS_FOLDER)],
    key=lambda x: float(re.findall("(\d+)", os.path.basename(x))[0]),
)

df_dict = {}
for i, (paper_path, keys_path) in enumerate(zip(papers_paths, keys_paths)):
    keys = pd.read_csv(keys_path, header=None)
    keys_list = keys.iloc[:,0].to_list()

    with open(paper_path) as f:
        lines = f.readlines()

    indices = []
    for i, line in enumerate(lines):
        if "--A".casefold() in line.casefold():
            indices.append(i)
        if "--B".casefold() in line.casefold():
            indices.append(i)
            break
    abstract = " ".join(lines[indices[0] + 1 : indices[-1]])

    df_dict[os.path.basename(keys_path).split(".")[0]]={"abstract":abstract, "keywords":keys_list}

df = pd.DataFrame.from_dict(df_dict)
df.to_csv(os.path.join(ROOT_DIR, "ai", "data", "labeled_abstracts_df.csv"), index=True)