# -*- coding: utf-8 -*-

"""
Script to build a vocabulary from the A-OKVQA dataset

This script will create a vocabulary from the A-OKVQA dataset. The vocabulary
will contain all the unique words in the dataset. This script was inspired by
the script given in https://github.com/allenai/aokvqa
"""

__author__ = "Mir Sazzat Hossain"

import argparse
import os
from collections import Counter

from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='dataset/vocab.txt',
                        help='Path to save the vocabulary')

    args = parser.parse_args()

    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    # Load dataset from HuggingFace
    train_dataset = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    val_dataset = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")

    # Create a vocabulary
    vocab = []
    all_choices = Counter()
    direct_answers = Counter()

    for i in train_dataset:
        vocab.append(i['choices'][i['correct_choice_idx']])
        all_choices.update(i['choices'])
        if isinstance(i['direct_answers'], str):
            i['direct_answers'] = eval(i['direct_answers'])
        direct_answers.update(set(i['direct_answers']))

    vocab += [k for k, v in all_choices.items() if v >= 3]
    vocab += [k for k, v in direct_answers.items() if v >= 3]

    vocab = sorted(set(vocab))
    print(f"Vocab size: {len(vocab)}")

    # Save the vocabulary
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))

    # Check coverage of the validation set
    is_matched = [v['choices'][v['correct_choice_idx']]
                  in vocab for v in val_dataset]
    coverage = sum(is_matched) / len(is_matched) * 100
    print(f'Validation set coverage: {coverage:.2f}%')
