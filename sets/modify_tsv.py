import pandas as pd

old_tsv_path = "sets/vgg-test.tsv.old"
new_tsv_path = "sets/vgg3-test.tsv"
df_list = pd.read_csv(old_tsv_path, sep="\t", dtype={"id": str}).to_dict("records")

for record in df_list:
    id = record["id"]
    start_time = int(id.split("_")[-1])
    name = "_".join(id.split("_")[:-1])
    start_time = start_time * 1000
    end_time = start_time + 10_000
    new_id = f"{name}_{start_time}_{end_time}"
    record["id"] = new_id

df = pd.DataFrame(df_list)
df.to_csv(new_tsv_path, sep="\t", index=False)
print(f"Saved to {new_tsv_path}")
