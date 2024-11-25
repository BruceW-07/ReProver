"""Data module for the tactic generator."""

import os
import json
import pickle
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ByT5Tokenizer

from common import (
    Batch,
    Corpus,
    Example,
    remove_marks,
    format_augmented_state,
)


class GeneratorDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        corpus: Corpus,
        preds: List[Dict[str, Any]],
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        tokenizer: ByT5Tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.preds = preds
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = self._load_data(data_path)

    # 加载数据, 返回以 tactic 为条目的数据列表
    # 包含 url, commit, file_path, full_name, state, tactic
    def _load_data(self, data_path: str) -> List[Example]:
        data = []
        for thm in tqdm(json.load(open(data_path))):
            for tac in thm["traced_tactics"]:
                tactic = remove_marks(tac["tactic"])
                data.append(
                    {
                        "url": thm["url"],
                        "commit": thm["commit"],
                        "file_path": thm["file_path"],
                        "full_name": thm["full_name"],
                        "state": tac["state_before"],
                        "tactic": tactic,
                    }
                )

        logger.info(f"{len(data)} examples loaded")
        return data

    def __len__(self) -> int:
        return len(self.data)

    # 提供单个数据样本的访问
    # 返回的是一个 Example 对象, 包含了拼接了 retrieve 到的 premises 的 state
    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]

        if self.preds is not None:
            file_path = ex["file_path"]
            pred = self.preds[(file_path, ex["full_name"], ex["state"])]
            # 将 state 和 retrieve 到的 premises 进行拼接
            # 在训练时以 self.p_drop 的概率舍弃掉一些 premises
            ex["state"] = format_augmented_state(
                ex["state"],
                pred["retrieved_premises"],
                self.max_inp_seq_len,
                self.p_drop if self.is_train else 0.0,
            )

        ex["state"] = remove_marks(ex["state"])
        return ex

    # 对数据进行标记化 (tokenize), 并将多个数据样本组合成一个 batch
    def collate(self, examples: List[Example]) -> Batch:
        # 对状态进行标记化
        state = [ex["state"] for ex in examples]
        tokenized_state = self.tokenizer(
            state,
            # 在批处理中，所有序列都会被填充到该批次中最长序列的长度
            padding="longest",
            max_length=self.max_inp_seq_len,
            # 允许将超过 max_length 的序列进行截断
            truncation=True,
            # 指定返回 PyTorch 张量格式的数据。"pt" 是 PyTorch 的缩写，这表明该代码是在 PyTorch 框架下运行的
            return_tensors="pt",
        )

        # 对策略进行标记化
        tactic = [ex["tactic"] for ex in examples]
        tokenized_tactic = self.tokenizer(
            tactic,
            padding="longest",
            max_length=self.max_oup_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tactic_ids = tokenized_tactic.input_ids
        # 将 padding 变成 -100
        tactic_ids[tactic_ids == self.tokenizer.pad_token_id] = -100

        batch = {}
        batch["state"] = state
        batch["state_ids"] = tokenized_state.input_ids
        # attention_mask: 1 表示该位置是实际的 token; 0 表示该位置是填充的 padding token
        batch["state_mask"] = tokenized_state.attention_mask
        batch["tactic"] = tactic
        batch["tactic_ids"] = tactic_ids
        batch["tactic_mask"] = tokenized_tactic.attention_mask

        # Copy other fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        num_workers: int,
        corpus_path: Optional[str] = None,
        preds_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        if corpus_path is not None:
            self.corpus = Corpus(corpus_path)
        else:
            self.corpus = None
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        # 并行进程的数量
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # self.preds 是一个字典，键为 (file_path, full_name, state)，值为 retrieve 到的数据
        if preds_path is None:
            logger.info("Without retrieval data")
            self.preds = None
        else:
            logger.info("With retrieval data")
            self.preds = {}
            for pred in pickle.load(open(preds_path, "rb")):
                ctx = pred["context"]
                self.preds[ctx.path, ctx.theorem_full_name, ctx.state] = pred
                

    def prepare_data(self) -> None:
        pass

    # 根据不同的 stage 使用 GeneratorDataset 加载数据
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = GeneratorDataset(
                os.path.join(self.data_path, "train.json"),
                self.corpus,
                self.preds,
                self.max_inp_seq_len,
                self.max_oup_seq_len,
                self.p_drop,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = GeneratorDataset(
                os.path.join(self.data_path, "val.json"),
                self.corpus,
                self.preds,
                self.max_inp_seq_len,
                self.max_oup_seq_len,
                self.p_drop,
                self.tokenizer,
                is_train=False,
            )

    # 返回 pytorch 的DataLoader 对象
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
