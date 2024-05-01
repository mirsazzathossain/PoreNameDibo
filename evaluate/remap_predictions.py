# -*- coding: utf-8 -*-

"""
Script to prepare predictions for evaluation in the A-OKVQA dataset. This script
was inspired by the script given in: https://github.com/allenai/aokvqa.
"""

import argparse
import json
import pathlib

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm


def map_to_choices(dataset: dict,
                   predictions: dict, device: str = 'cuda') -> dict:
    """
    Maps the predictions to the choices in the dataset.

    :param dataset: The dataset.
    :type dataset: dict
    :param predictions: The predictions.
    :type predictions: dict
    :param device: The device to use for the model.
    :type device: str

    :return: The mapped predictions.
    :rtype: dict
    """
    if isinstance(dataset, list):
        dataset = {dataset[i]['question_id']: dataset[i]
                   for i in range(len(dataset))}

    if all([p in dataset[q]['choices'] for q, p in predictions.items()]):
        return predictions

    model = SentenceTransformer(
        'sentence-transformers/average_word_embeddings_glove.6B.300d')
    model.to(device)
    for q in tqdm(predictions.keys()):
        choices = dataset[q]['choices']
        if predictions[q] not in choices:
            choice_embeddings = model.encode(
                [predictions[q]] + choices, convert_to_tensor=True)
            a_idx = cos_sim(
                choice_embeddings[0], choice_embeddings[1:]).argmax().item()
            predictions[q] = choices[a_idx]

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aokvqa-dir', type=pathlib.Path,
                        required=True, dest='aokvqa_dir')
    parser.add_argument('--split', type=str,
                        choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--pred', type=argparse.FileType('r'),
                        required=True, dest='prediction_file')
    parser.add_argument('--out', type=argparse.FileType('w'),
                        required=True, dest='output_file')
    args = parser.parse_args()

    dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=args.split)
    predictions = json.load(args.prediction_file)
    predictions = map_to_choices(dataset, predictions)

    json.dump(predictions, args.output_file)
