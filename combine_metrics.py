import json
import csv
from pathlib import Path
from collections import defaultdict


METRIC_FILENAME = "output_metrics.json"
EXP = "controlnet_small_44k_m_to_l_ec_sum_post_no_preconv"


def collect_metrics(json_dir):
    metrics_data = []
    for json_file in Path(json_dir).rglob("output_metrics_avg.json"):
        with open(json_file, "r") as f:
            metrics = json.load(f)
            parent_dir = json_file.parent.name
            metrics["ID"] = parent_dir
            metrics_data.append(metrics)
    return metrics_data


def write_metrics_to_csv(metrics_data, output_csv):
    if not metrics_data:
        print("No metrics data found.")
        return

    # Get the keys from the first dictionary as the CSV headers
    headers = metrics_data[0].keys()

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for data in metrics_data:
            writer.writerow(data)


def read_metrics(json_files):
    metrics = defaultdict(list)
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                metrics[key].append(value)
    return metrics


def average_metrics(metrics):
    averaged_metrics = {}
    for key, values in metrics.items():
        averaged_metrics[key] = sum(values) / len(values)
    return averaged_metrics


def main():
    json_dir = Path("path/to/json/files")  # Replace with the path to your JSON files
    json_files = list(json_dir.glob("**/output_metrics.json"))

    metrics = read_metrics(json_files)
    averaged_metrics = average_metrics(metrics)

    # Print or save the averaged metrics
    print(json.dumps(averaged_metrics, indent=4))


if __name__ == "__main__":
    json_dir = f"./output/{EXP}"  # Replace with the path to your JSON files
    output_csv = f"./output/output_metrics_avg.csv"  # Replace with the desired output CSV file name
    output_json = f"./output/{EXP}/output_metrics_avg.json"  # Replace with the desired output CSV file name

    # average JSONs
    # json_files = list(Path(json_dir).rglob(METRIC_FILENAME))
    # metrics = read_metrics(json_files)
    # averaged_metrics = average_metrics(metrics)
    # json_file = Path(output_json)
    # with open(json_file, "w") as f:
    #     json.dump(averaged_metrics, f, indent=4)

    # write averaged JSONs to CSV
    metrics_data = collect_metrics("./output")
    write_metrics_to_csv(metrics_data, output_csv)
