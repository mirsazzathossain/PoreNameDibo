# -*- coding: utf-8 -*-

"""
Script to extract Faster R-CNN features from the A-OKVQA dataset

This script will extract Faster R-CNN features from the A-OKVQA dataset. The
Faster R-CNN features will be extracted using the HuggingFace Transformers
library. This script was inspired by the script given in:
    1. https://github.com/allenai/aokvqa
    2. https://github.com/huggingface/transformers/blob/main/examples/research_projects/visual_bert/demo.ipynb
"""

__author__ = "Mir Sazzat Hossain"


import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from VisualBert.modeling_frcnn import GeneralizedRCNN
from VisualBert.processing_image import Preprocess
from VisualBert.utils import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['train', 'validation', 'test'],
                        default='train', help='Split to extract Faster R-CNN features')
    parser.add_argument('--output_file', type=str, default='dataset/frcnn_features.pt',
                        help='Path to save the Faster R-CNN features')
    args = parser.parse_args()

    assert args.output_file.endswith('.pt'), 'Output file must be a .pt file'

    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    # Load dataset from HuggingFace
    dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=args.split)

    # Load Faster R-CNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frcnn_config = Config.from_pretrained('unc-nlp/frcnn-vg-finetuned')
    frcnn_config.MODEL.DEVICE = device
    frcnn = GeneralizedRCNN.from_pretrained(
        'unc-nlp/frcnn-vg-finetuned', config=frcnn_config)
    preprocessor = Preprocess(frcnn_config)

    # Extract Faster R-CNN features
    with torch.no_grad():
        embeddings = {}

        for d in tqdm(dataset, desc='Extracting Faster R-CNN features'):
            image = d['image'].convert('RGB')
            image = torch.as_tensor(np.array(image))

            image, sizes, scales_yx = preprocessor([image])

            frcnn_output = frcnn(
                image, sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=frcnn_config.max_detections,
                return_tensors="pt"
            )

            visual_embeds = frcnn_output.get(
                "roi_features").detach().cpu().squeeze()
            visual_token_type_ids = torch.ones(
                visual_embeds.shape[:-1], dtype=torch.long).squeeze()
            visual_attention_mask = torch.ones(
                visual_embeds.shape[:-1], dtype=torch.long).squeeze()

            embeddings[d['question_id']] = {
                'visual_embeds': visual_embeds,
                'visual_token_type_ids': visual_token_type_ids,
                'visual_attention_mask': visual_attention_mask
            }

        torch.save(embeddings, args.output_file)
