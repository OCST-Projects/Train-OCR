pip freeze > requirements.txt

--extra-index-url https://download.pytorch.org/whl/cu121

python create_lmdb_dataset.py --inputPath dataset/train --gtFile dataset/train/labels.csv --outputPath lmdb_data/train
python create_lmdb_dataset.py --inputPath dataset/val --gtFile dataset/val/labels.csv --outputPath lmdb_data/val

python train.py --train_data /lmdb_data/train --valid_data /lmdb_data/val --select_data /lmdb_data --batch_ratio 0.5 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC
