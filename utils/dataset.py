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


# class AokvqaEmbeddingsDataModule(pl.LightningDataModule):
#     """Data module for AOKVQA embeddings"""

#     def __init__(
#             self,
#             train_features: list[str],
#             val_features: list[str],
#             vocab: list[str],
#             batch_size: int,
#             num_workers: int) -> None:
#         """
#         Initialize the data module

#         :param train_features: The train features
#         :type train_features: list[str]
#         :param val_features: The validation features
#         :type val_features: list[str]
#         :param vocab: The vocabulary of the dataset
#         :type vocab: list[str]
#         :param batch_size: The batch size
#         :type batch_size: int
#         :param num_workers: The number of workers
#         :type num_workers: int
#         """
#         super().__init__()
#         self.train_features = train_features
#         self.val_features = val_features
#         self.vocab = vocab
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#     def setup(self, stage: str = None) -> None:
#         """
#         Setup the data module

#         :param stage: The stage of the data module
#         :type stage: str
#         """
#         if stage == 'fit' or stage is None:
#             self.train_dataset = AokvqaEmbeddingsDataset(
#                 'train', self.train_features[0], self.train_features[1], self.vocab)
#             self.val_dataset = AokvqaEmbeddingsDataset(
#                 'validation', self.val_features[0], self.val_features[1], self.vocab)

#     def train_dataloader(self) -> DataLoader:
#         """
#         Get the train dataloader

#         :return: The train dataloader
#         :rtype: DataLoader
#         """
#         return DataLoader(
#             self.train_dataset, batch_size=self.batch_size,
#             num_workers=int(0.8 * self.num_workers), shuffle=True)

#     def val_dataloader(self) -> DataLoader:
#         """
#         Get the validation dataloader

#         :return: The validation dataloader
#         :rtype: DataLoader
#         """
#         return DataLoader(
#             self.val_dataset, batch_size=self.batch_size,
#             num_workers=int(0.2 * self.num_workers), shuffle=False)


if __name__ == "__main__":
    vocab = []
    with open('dataset/vocab.csv', 'r') as f:
        for line in f:
            vocab.append(line.strip())

    dataset = AokvqaEmbeddingsDataset(
        split='train',
        rcnn_features='dataset/train/frcnn_features.pt',
        bert_features='dataset/train/bert_features.pt',
        vocab=vocab)
    for i in range(len(dataset)):
        e, a = dataset[i]
        print(a.sum())
        print(a)

    # dm = AokvqaEmbeddingsDataModule(
    #     train_features=['dataset/train/frcnn_features.pt',
    #                     'dataset/train/bert_features.pt'],
    #     val_features=['dataset/validation/frcnn_features.pt',
    #                   'dataset/validation/bert_features.pt'],
    #     objective='classification',
    #     vocab=vocab,
    #     vocab_features=None,
    #     batch_size=4,
    #     num_workers=torch.get_num_threads() - 1)

    # dm.setup()
    # train_loader = dm.train_dataloader()
    # val_loader = dm.val_dataloader()

    # for batch in train_loader:
    #     x, y = batch
    #     for key in x:
    #         print(key, x[key].shape)
    #         inputs_embeds = x['inputs_embeds']
    #         input_shape = inputs_embeds.size()[:-1]
    #         print(input_shape)
    #         batch_size, seq_length = input_shape
    #         print(batch_size, seq_length)

    #     print(y.shape)

    # for batch in val_loader:
    #     x, y = batch

    #     for key in x:
    #         print(key, x[key].shape)

    #     print(y.shape)
