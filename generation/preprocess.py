"""Script for preprocess state-tactic pairs into the format required by [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)."""

import os
import json
import random
import argparse
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/leandojo_benchmark_4/random",
    )
    parser.add_argument("--dst-path", type=str, default="state_tactic_pairs")
    args = parser.parse_args()
    logger.info(args)

    for split in ("train", "val"):
        data_path = os.path.join(args.data_path, f"{split}.json")
        pairs = []
        # 遍历 thm 中的 traced_tactics, 将 state_before 和 tactic 作为输入输出对
        for thm in json.load(open(data_path)):
            for tac in thm["traced_tactics"]:
                # 输入为 state_before, 输出为 tactic
                pairs.append({"state": tac["state_before"], "output": tac["tactic"]})
        logger.info(f"Read {len(pairs)} state-tactic paris from {data_path}")

        # 打乱数据
        random.shuffle(pairs)
        data = [
            {
                "instruction": f"[GOAL]\n{pair['state']}\n[PROOFSTEP]\n",
                "input": "",
                "output": pair["output"],
            }
            for pair in pairs
        ]
        logger.info(data[0])
        dst_path = args.dst_path + f"_{split}.json"
        json.dump(data, open(dst_path, "wt"))
        # 数据默认存储在当前文件夹下的 state_tactic_pairs_{split}.json
        logger.info(f"Preprocessed data saved to {dst_path}")


if __name__ == "__main__":
    main()
