# -*- coding: utf-8 -*-

"""
Script to extract BERT features from the A-OKVQA dataset

This script will extract BERT features from the A-OKVQA dataset. The BERT
features will be extracted using the HuggingFace Transformers library. This
script was inspired by the script given in https://github.com/allenai/aokvqa.
"""

__author__ = "Mir Sazzat Hossain"


import argparse
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling of the BERT output

    This function will perform mean pooling of the BERT output. The mean pooling
    is performed by taking the sum of the token embeddings and dividing it by
    the sum of the attention mask.

    :param model_output: BERT output
    :type model_output: torch.Tensor
    :param attention_mask: Attention mask
    :type attention_mask: torch.Tensor

    :return: Mean pooled BERT output
    :rtype: torch.Tensor
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_embeddings / torch.clamp(sum_mask, min=1e-9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['train', 'validation', 'test'],
                        default='train', help='Split to extract BERT features')
    parser.add_argument('--output_file', type=str, default='dataset/bert_features.pt',
                        help='Path to save the BERT features')
    args = parser.parse_args()

    assert args.output_file.endswith('.pt'), 'Output file must be a .pt file'

    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    # Load dataset from HuggingFace
    dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=args.split)

    # Load BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained(
        'sentence-transformers/bert-base-nli-mean-tokens')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Extract BERT features
    with torch.no_grad():
        embeddings = {}

        for d in tqdm(dataset, desc='Extracting BERT features'):
            question = d['question']

            encoded_input = tokenizer(
                [question], return_tensors='pt', padding='max_length',
                truncation=True, max_length=128)

            embeddings[d['question_id']] = {
                k: v for k, v in encoded_input.items()}
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            e = mean_pooling(model(**encoded_input),
                             encoded_input['attention_mask'])

            embeddings[d['question_id']]['inputs_embeds'] = e.detach().cpu()

        torch.save(embeddings, args.output_file)
