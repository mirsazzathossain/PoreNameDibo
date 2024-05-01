# -*- coding: utf-8 -*-

"""
Script to prepare predictions for evaluation in the A-OKVQA dataset. This script
was inspired by the script given in: https://github.com/allenai/aokvqa.
"""

__author__ = "Mir Sazzat Hossain"

import argparse
import glob
import json
import pathlib

from datasets import load_dataset


def eval_aokvqa(dataset: dict, preds: dict, 
                multiple_choice: bool = False, strict: bool = True) -> float:
    """
    Evaluate predictions on the A-OKVQA dataset.

    :param dataset: A dictionary containing the A-OKVQA dataset.
    :type dataset: dict
    :param preds: A dictionary containing the predictions.
    :type preds: dict
    :param multiple_choice: A boolean indicating whether the predictions are 
                            for multiple choice questions.
    :type multiple_choice: bool
    :param strict: A boolean indicating whether to strictly enforce that all 
                    questions in the dataset are in the predictions.
    :type strict: bool

    :return: The accuracy of the predictions.
    :rtype: float
    """

    if isinstance(dataset, list):
        dataset = {dataset[i]['question_id']: dataset[i]
                   for i in range(len(dataset))}

    if multiple_choice is False:
        dataset = {k: v for k, v in dataset.items(
        ) if v['difficult_direct_answer'] is False}

    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    acc = []

    for v in dataset.keys():
        if v not in preds.keys():
            acc.append(0.0)
            continue

        pred = preds[v]
        choices = dataset[v]['choices']
        direct_answers = dataset[v]['direct_answers']

        # Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
            correct_choice_idx = dataset[v]['correct_choice_idx']
            acc.append(float(pred == choices[correct_choice_idx]))
            
        # Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) * 100

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aokvqa-dir', type=pathlib.Path,
                        required=True, dest='aokvqa_dir')
    parser.add_argument('--split', type=str,
                        choices=['train', 'val'], required=True)
    parser.add_argument('--preds', type=str, required=True,
                        dest='prediction_files')
    args = parser.parse_args()

    dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=args.split)

    for prediction_file in glob.glob(args.prediction_files):
        predictions = json.load(open(prediction_file, 'r', encoding='utf-8'))

        # Multiple choice
        mc_predictions = {}

        for q in predictions.keys():
            if 'multiple_choice' in predictions[q].keys():
                mc_predictions[q] = predictions[q]['multiple_choice']

        if mc_predictions:
            mc_acc = eval_aokvqa(
                dataset,
                mc_predictions,
                multiple_choice=True,
                strict=False
            )
            print(prediction_file, 'MC', mc_acc)

        # Direct Answer
        da_predictions = {}

        for q in predictions.keys():
            if 'direct_answer' in predictions[q].keys():
                da_predictions[q] = predictions[q]['direct_answer']

        if da_predictions:
            da_acc = eval_aokvqa(
                dataset,
                da_predictions,
                multiple_choice=False,
                strict=False
            )
            print(prediction_file, 'DA', da_acc)
