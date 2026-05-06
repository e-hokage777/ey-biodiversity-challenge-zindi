import joblib
from argparse import ArgumentParser
import pandas as pd
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--test-path", type=str, default="data/test-with-satellite-features.csv"
    )
    parser.add_argument("--pipeline-path", type=str, required=True)
    # parser.add_argument("--save-name", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_df = pd.read_csv(args.test_path)
    pipeline = joblib.load(args.pipeline_path)

    if not os.path.exists("submissions"):
        os.makedirs("submissions")

    save_path = os.path.join(
        "submissions", os.path.basename(os.path.dirname(args.pipeline_path)) + ".csv"
    )

    ids = pd.read_csv("data/Test.csv")["ID"]

    predictions = pipeline.predict(test_df)
    sub = pd.DataFrame({"ID": ids, "Occurrence Status": predictions})
    sub.to_csv(save_path, index=False)
