#!/bin/bash


# NOTE:
# 1) Not using distorted images for now
# 2) bekern and kern results in the same sequence after the parser
# so we only need to run one of them -> kern

for input_modality in image audio; do
    for train_ds in grandstaff beethoven chopin hummel joplin mozart scarlatti-d; do
        python -u train.py --ds_name $train_ds --krn_encoding kern --input_modality $input_modality --attn_window 100 --epochs 300 --patience 5 --batch_size 1 
        for test_ds in grandstaff beethoven chopin hummel joplin mozart scarlatti-d; do
            if [ $train_ds != $test_ds ]; then
                python -u test.py --ds_name $test_ds --krn_encoding kern --input_modality $input_modality --checkpoint_path weights/$train_ds/$input_modality\_kern.ckpt
            fi
        done
    done
done
 