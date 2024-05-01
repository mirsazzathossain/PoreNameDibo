python data_scripts/build_vocab.py --output_file dataset/vocab.csv
python data_scripts/extract_bert_feature.py --split train --output_file dataset/train/bert_features.pt
python data_scripts/extract_bert_feature.py --split validation --output_file dataset/validation/bert_features.pt
python data_scripts/extract_bert_feature.py --split test --output_file dataset/test/bert_features.pt
python data_scripts/extract_frcnn_feature.py --split train --output_file dataset/train/frcnn_features.pt
python data_scripts/extract_frcnn_feature.py --split validation --output_file dataset/validation/frcnn_features.pt
python data_scripts/extract_frcnn_feature.py --split test --output_file dataset/test/frcnn_features.pt
