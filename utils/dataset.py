# -*- coding: utf-8 -*-

"""
Script to build the A-OKVQA dataset.

This script contains the dataset class for A-OKVQA embeddings and the data 
module for A-OKVQA embeddings. The code is adapted from the following source:
https://github.com/allenai/aokvqa
"""

__author__ = "Mir Sazzat Hossain"


import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class AokvqaEmbeddingsDataset(Dataset):
    """
    Dataset class for AOKVQA embeddings
    """

    def __init__(
            self,
            split: str,
            rcnn_features: str,
            bert_features: str,
            vocab: list[str]) -> None:
        """
        Initialize the dataset

        :param split: The split of the dataset
        :type split: str
        :param rcnn_features: Path to the RCNN features
        :type rcnn_features: str
        :param bert_features: Path to the BERT features
        :type bert_features: str
        :param vocab: The vocabulary of the dataset
        :type vocab: list[str]
        """
        self.split = split
        self.rcnn_features = rcnn_features
        self.bert_features = bert_features
        self.vocab = vocab

        rcnn_embeddings = torch.load(self.rcnn_features)
        bert_embeddings = torch.load(self.bert_features)

        self.vocab_len = len(self.vocab)

        dataset = load_dataset('HuggingFaceM4/A-OKVQA', split=self.split)
        self.embeddings = []
        self.answers = []

        for o in dataset:
            if isinstance(o['direct_answers'], str):
                o['direct_answers'] = eval(o['direct_answers'])
            correct_answers = set(
                [o['choices'][o['correct_choice_idx']]] + o['direct_answers'])
            correct_answers = [vocab.index(a)
                               for a in correct_answers if a in vocab]
            if len(correct_answers) == 0:
                continue
            self.answers.append(correct_answers)

            q = o['question_id']

            e = bert_embeddings[q]
            e.update(rcnn_embeddings[q])
            e = {k: v.squeeze() if k != 'inputs_embeds' else v for k, v in e.items()}
            self.embeddings.append(e)

    def __len__(self) -> int:
        """
        Get the length of the dataset

        :return: The length of the dataset
        :rtype: int
        """
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset

        :param idx: The index of the item
        :type idx: int
        :return: The item from the dataset
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        e = self.embeddings[idx]
        a = self.answers[idx]

        a = torch.sum(
            F.one_hot(
                torch.tensor(a), num_classes=self.vocab_len), dim=0)
        # a = a.float()

        return e, a
