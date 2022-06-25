#!/bin/sh
pwd
nvidia-smi
echo "FINE!"
#python3 ./covergan_train.py --emotions ./emotions.json --covers ./clean_covers/ --audio ./audio --epochs 50000
#python3 ./covergan_train.py --emotions ./small_emotions.json --covers ./small_clean_covers/ --audio ./small_audio --checkpoint_root ./small_checkpoint --epochs 50000
#python3 ./covergan_train.py --train_dir ./dataset_demo_4 --emotions ./demo_emotions.json --epochs 50000 --display_steps 300
#python3 ./covergan_train.py --train_dir ./dataset_emoji_4 --emotions ./emotions.json --epochs 50000 --display_steps 100
#python3 ./covergan_train.py --train_dir ./dataset_emoji_52 --emotions ./emotions.json --epochs 50000 --display_steps 300
python3 ./covergan_train.py --train_dir ./dataset_full_covers --emotions ./emotions.json --epochs 50000 --display_steps 100 --backup_epochs 20 --gen_lr 0.0001 --disc_lr 0.0004
#python3 ./colorer_train.py --train_dir ./dataset_full_covers --emotions ./emotions.json --epochs 50000 --display_steps 500 --backup_epochs 50