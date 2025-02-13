import pandas as pd

tsv_train = {"id": [], "label": []}
tsv_val = {"id": [], "label": []}
tsv_test = {"id": [], "label": []}
metadata_file = "/home/hdd/ilpo/datasets/AVSSemantic/Single-source/s4_meta_data.csv"

md_list = pd.read_csv(metadata_file, sep=",").to_dict("records")

for record in md_list:
    id = record["name"]
    label = record["category"]
    split = record["split"]
    if split == "train":
        tsv = tsv_train
    elif split == "val":
        tsv = tsv_val
    elif split == "test":
        tsv = tsv_test
    else:
        raise ValueError(f"Unknown split: {split}")
    tsv["id"].append(id)
    tsv["label"].append(label)

for i, tsv in enumerate([tsv_train, tsv_test, tsv_val]):
    if i == 0:
        split = "train"
    elif i == 1:
        split = "test"
    else:
        split = "val"

    df = pd.DataFrame(tsv)
    df.to_csv(f"sets/avs-{split}.tsv", sep="\t", index=False)
