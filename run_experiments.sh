#!/bin/bash


# NOTE:
# Not using distorted images for now

for input_modality in audio image; do
    for krn_encoding in bekern kern; do
        for train_ds in grandstaff beethoven chopin hummel joplin mozart scarlatti-d; do
            python -u train.py --ds_name $train_ds --krn_encoding $krn_encoding --input_modality $input_modality --attn_window 100 --epochs 300 --patience 5 --batch_size 1 
            for test_ds in grandstaff beethoven chopin hummel joplin mozart scarlatti-d; do
                if [ $train_ds != $test_ds ]; then
                    python -u test.py --ds_name $test_ds --krn_encoding $krn_encoding --input_modality $input_modality --checkpoint_path weights/$train_ds/$input_modality\_$krn_encoding.ckpt
                fi
            done
        done
    done
done
 