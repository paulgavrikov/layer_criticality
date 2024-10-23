from glob import glob
import pandas as pd
import torch
from multiprocessing import Pool
import argparse
import os


def read_weight_reset_pt(pt_file):
    try:
        exp_result = torch.load(pt_file, map_location='cpu') 

        for d in exp_result["trials"]:
            if "logits" in d:
                del d["logits"]

        df = pd.DataFrame(exp_result["trials"])
        df["recovered_acc"] = df["top1_acc"].apply(lambda x: x / df.query("layer == 'original'")["top1_acc"].item())
        df["model_id"] = exp_result["args"]["model"]
        df = df.query("layer != 'original'") 
    except Exception as e:
        print(f"Error in {pt_file}: {e}")
        return pd.DataFrame()
    return  df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_path", type=str, required=True, help="Path to the directory containing the .pt files", default="./measurements/weight_reset")
    argparser.add_argument("--output_file", type=str, required=True, help="Path to the output csv file", default="../results/results.csv")
    argparser.add_argument("--n_worker", type=int, default=16)

    args = argparser.parse_args()

    root_dir = os.path.join(args.input_path, "*.pt")

    with Pool(args.n_worker) as p:
        df_weight_reset = pd.concat(p.imap(read_weight_reset_pt, glob(root_dir)))

    # create output directory if necessary
    parent_dir = os.path.dirname(args.output_file)
    if parent_dir and parent_dir != "":
        os.makedirs(parent_dir, exist_ok=True)

    df_weight_reset.to_csv(args.output_file, index=False)
