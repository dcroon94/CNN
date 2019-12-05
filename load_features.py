#Date:      5-dec-2019
#Developer: Dennis Croon


import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    num_lines = sum(1 for line in open('features.jsonl'))
    with jsonlines.open("features.jsonl") as jsonl_fh:
        train, test = [], []

        for obj in tqdm(jsonl_fh, desc="Loading Features", total=num_lines):
            if obj["dataset"] == "train":
                train.append(obj)
            elif obj["dataset"] == "test":
                test.append(obj)

    train_df = pd.DataFrame.from_records(train)
    test_df = pd.DataFrame.from_records(test)

    print(train_df.head())
    print(test_df.head())

if __name__ == "__main__":
    main()