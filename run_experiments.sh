#!/bin/bash


# NOTE:
# 1) Using distorted images and clean audios for training and testing
# 2) bekern and kern results in the same sequence after the parser
# so we only need to run one of them -> kern

for input_modality in image audio; do
    for train_ds in grandstaff beethoven chopin hummel joplin mozart scarlatti-d; do
        python -u train.py --ds_name $train_ds --krn_encoding kern --input_modality $input_modality --attn_window 100 --epochs 300 --patience 5 --batch_size 1  --use_distorted_images
        for test_ds in grandstaff beethoven chopin hummel joplin mozart scarlatti-d; do
            if [ $train_ds != $test_ds ]; then
                if [ $input_modality == "image" ]; then
                    checkpoint_path=weights/$train_ds/image_distorted_kern.ckpt
                else
                    checkpoint_path=weights/$train_ds/audio_kern.ckpt
                python -u test.py --ds_name $test_ds --krn_encoding kern --input_modality $input_modality --checkpoint_path $checkpoint_path  --use_distorted_images
            fi
        done
    done
done
 