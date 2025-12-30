#!/bin/bash

# NOTE:
# 1) Using distorted images and clean audios for training and testing
# 2) bekern and kern results in the same sequence after the parser
# so we only need to run one of them -> kern

############################## UNIMODAL AND MULTIMODAL TRANSFORMER EXPERIMENTS:

for input_modality in image audio both; do
    for mixer_type in concat attn_img attn_audio attn_both; do
        for train_ds in joplin mozart beethoven chopin scarlatti-d grandstaff; do
            python -u src/train.py --ds_name $train_ds --krn_encoding kern --input_modality $input_modality --mixer_type $mixer_type --attn_window 100 --epochs 300 --patience 5 --batch_size 1 --use_distorted_images
            for test_ds in grandstaff beethoven chopin hummel joplin mozart scarlatti-d; do
                if [ $train_ds != $test_ds ]; then
                    if [ $input_modality == "image" ]; then
                        checkpoint_path=weights/$train_ds/image_distorted_kern.ckpt
                    elif [ $input_modality == "audio" ]; then
                        checkpoint_path=weights/$train_ds/audio_kern.ckpt
                    else
                        checkpoint_path=weights/$train_ds/both_"$mixer_type"_kern.ckpt
                    fi
                    python -u src/test.py --ds_name $test_ds --krn_encoding kern --input_modality $input_modality --checkpoint_path $checkpoint_path --use_distorted_images
                fi
            done
        done
    done
done

############################## LATE-FUSION SMITH-WATERMAN EXPERIMENTS:

match=(2 10 20 5)
mismatch=( -1 5 10 2 )
gap_penalty=( -1 -2 -4 -1 )

for i in "${!match[@]}"; do
    m="${match[$i]}"
    mm="${mismatch[$i]}"
    g="${gap_penalty[$i]}"

    for test_ds in hummel joplin mozart beethoven chopin scarlatti-d grandstaff; do
        for image_ds in joplin mozart beethoven chopin scarlatti-d; do
            for audio_ds in joplin mozart beethoven chopin scarlatti-d; do
                image_checkpoint_path=weights/$image_ds/image_distorted_kern.ckpt
                audio_checkpoint_path=weights/$audio_ds/audio_kern.ckpt

                python src/multimodal/smith_waterman/test.py \
                    --match "$m" \
                    --mismatch "$mm" \
                    --gap_penalty "$g" \
                    --ds_name "$test_ds" \
                    --krn_encoding kern \
                    --image_checkpoint_path "$image_checkpoint_path" \
                    --audio_checkpoint_path "$audio_checkpoint_path" \
                    --use_distorted_images
            done
        done
    done
done


############################## LATE-FUSION WEIGHTED AVERAGE EXPERIMENTS:

alpha=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for i in "${!alpha[@]}"; do
    a="${alpha[$i]}"

    for test_ds in hummel joplin mozart beethoven chopin scarlatti-d grandstaff; do
        for image_ds in joplin mozart beethoven chopin scarlatti-d; do
            for audio_ds in joplin mozart beethoven chopin scarlatti-d; do
                image_checkpoint_path=weights/$image_ds/image_distorted_kern.ckpt
                audio_checkpoint_path=weights/$audio_ds/audio_kern.ckpt

                python src/multimodal/weighted_multimodal/test.py \
                    --alpha "$a" \
                    --ds_name "$test_ds" \
                    --krn_encoding kern \
                    --image_checkpoint_path "$image_checkpoint_path" \
                    --audio_checkpoint_path "$audio_checkpoint_path" \
                    --use_distorted_images
            done
        done
    done
done
