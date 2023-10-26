#!/bin/sh

###################################################  TRAIN ALL MODELS ON ALL COMPOSERS:

mkdir -p results
for train_composer in hummel; do #beethoven chopin hummel joplin mozart scarlatti-d all; do
    mkdir -p results/$train_composer

    ################################ DAN:
    # Ligatures
    echo "Train Composer: $train_composer; Test Composer: $train_composer; Model: DAN_audio"
    python train.py --config config/audio2seq/grandstaff_$train_composer.gin > results/$train_composer/DAN_audio_$train_composer.txt

    echo "Train Composer: $train_composer; Test Composer: $train_composer; Model: DAN_image"
    python train.py --config config/image2seq/grandstaff_$train_composer.gin > results/$train_composer/DAN_image_$train_composer.txt
done

###################################################  TEST ALL MODELS AGAINST ALL COMPOSERS:

# NOTE:
# Hasta ahora solo hemos sacado el todos contra todos de los modelos DAN
# Para los modelos de tipo CTC, solo hemos enfretando el modelo "All" contra los otros compositores

# mkdir -p results
# for train_composer in All Mozart Beethoven Haydn; do
#     for test_composer in All Mozart Beethoven Haydn; do
#         mkdir -p results/$train_composer\_Ligatures

#         ################################ DAN:
#         # Ligatures
#         echo "Train Composer: $train_composer; Test Composer: $test_composer; Model: DAN_Ligatures"
#         python test_img2seq.py --config config/IMG2Seq/Ligatures/DAN_Quartets_$test_composer\_Ligatures.gin --checkpoint_path weights/$train_composer\_Ligatures/DAN_Ligatures.ckpt > results/$train_composer\_Ligatures/DAN_$train_composer\_$test_composer\_Ligatures.txt

#     done
# done
