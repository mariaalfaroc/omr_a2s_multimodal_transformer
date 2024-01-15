import os, shutil
import numpy as np
from pyMV2H.utils.mv2h import MV2H
from pyMV2H.utils.music import Music
from pyMV2H.metrics.mv2h import mv2h
from music21 import converter as converterm21
from pyMV2H.converter.midi_converter import MidiConverter as Converter

from utils.kern import CON_TOKEN, COC_TOKEN, COR_TOKEN

#### DEBUG
# CON_TOKEN = '<con>'
# COC_TOKEN = '<coc>'
# COR_TOKEN = '<cor>'
#### \DEBUG


def compute_metrics(y_true, y_pred, compute_mv2h=False):
    # ------------------------------- Sym-ER and Seq-ER:
    metrics = compute_ed_metrics(y_true=y_true, y_pred=y_pred)
    if compute_mv2h:
        # ------------------------------- MV2H:
        mv2h_dict = compute_mv2h_metrics(y_true=y_true, y_pred=y_pred)
        metrics.update(mv2h_dict)
    return metrics


###################################################################### METRICS UTILS:


def compute_ed_metrics(y_true, y_pred):
    def levenshtein(a, b):
        n, m = len(a), len(b)

        if n > m:
            a, b = b, a
            n, m = m, n

        current = range(n + 1)
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]

    ed_acc = 0
    length_acc = 0
    label_acc = 0
    for t, h in zip(y_true, y_pred):
        ed = levenshtein(t, h)
        ed_acc += ed
        length_acc += len(t)
        if ed > 0:
            label_acc += 1

    return {
        "sym-er": 100.0 * ed_acc / length_acc,
        "seq-er": 100.0 * label_acc / len(y_pred),
    }


def compute_mv2h_metrics(y_true, y_pred):
    def removeSpineTokens(in_file):
        tokensToRemove = ["*^\n", "*v\n"]

        with open(in_file) as fin:
            line = fin.readlines()

        for singleToken in tokensToRemove:
            try:
                indices = np.where(np.array(line) == singleToken)[0]
                line = list(np.delete(np.array(line), indices))
                with open(in_file, "w") as fout:
                    for element in line:
                        fout.write(element)
            except:
                pass
        return

    def krn2midi(in_file):
        removeSpineTokens(in_file)

        a = converterm21.parse(in_file).write("midi")
        midi_file = a.name
        shutil.copyfile(a, midi_file)
        os.remove(in_file)
        return midi_file

    def midi2txt(midi_file):
        txt_file = midi_file.replace("mid", "txt")
        converter = Converter(file=midi_file, output=txt_file)
        converter.convert_file()
        with open(txt_file, "r") as fin:
            f = [u.replace(".0", "") for u in fin.readlines()]
        with open(txt_file, "w") as fout:
            for u in f:
                fout.write(u)
        os.remove(midi_file)
        return txt_file

    """Polyphonic evaluation."""

    def eval_as_polyphonic():
        # Processing GT:
        reference_midi_file = krn2midi("gtKern.krn")

        # Processing prediction:
        predicted_midi_file = krn2midi("predKern.krn")

        # Converting to TXT:
        ### True:
        reference_txt_file = midi2txt(reference_midi_file)

        ### Pred:
        predicted_txt_file = midi2txt(predicted_midi_file)

        # Figures of merit:
        reference_file = Music.from_file(reference_txt_file)
        transcription_file = Music.from_file(predicted_txt_file)

        res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
        try:
            res_dict = mv2h(reference_file, transcription_file)
        except:
            pass

        os.remove(reference_txt_file)
        os.remove(predicted_txt_file)

        return res_dict

    def divide_voice(in_file, out_file, voice):
        bool_voiceExists = True

        # Opening file:
        with open(in_file) as fin:
            read_file = fin.readlines()

        # Read voice:
        try:
            voice = [u.split("\t")[voice].strip() for u in read_file]

            # Writing voice:
            with open(out_file, "w") as fout:
                for token in voice:
                    fout.write(token + "\n")
        except:
            bool_voiceExists = False

        return bool_voiceExists

    """Monophonic evaluation."""

    def eval_as_monophonic():
        global_res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)

        n_voices = 0
        eval_voices = True
        while eval_voices:
            res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)

            # Processing GT:
            gtVoice_exists = divide_voice("gtKern.krn", "gtVoiceKern.krn", n_voices)
            if gtVoice_exists:
                reference_midi_file = krn2midi("gtVoiceKern.krn")  # Converting to MIDI
                reference_txt_file = midi2txt(reference_midi_file)  # Converting to TXT
                reference_file = Music.from_file(reference_txt_file)

            # Processing prediction:
            predVoice_exists = divide_voice(
                "predKern.krn", "predVoiceKern.krn", n_voices
            )
            if predVoice_exists:
                predicted_midi_file = krn2midi(
                    "predVoiceKern.krn"
                )  # Converting to MIDI
                predicted_txt_file = midi2txt(predicted_midi_file)  # Converting to TXT
                transcription_file = Music.from_file(predicted_txt_file)

            if gtVoice_exists and predVoice_exists:
                n_voices += 1

                # Figures of merit:
                try:
                    res_dict = mv2h(reference_file, transcription_file)
                    global_res_dict.__multi_pitch__ += res_dict.multi_pitch
                    global_res_dict.__voice__ += res_dict.voice
                    global_res_dict.__meter__ += res_dict.meter
                    global_res_dict.__harmony__ += res_dict.harmony
                    global_res_dict.__note_value__ += res_dict.note_value
                except:
                    pass

            else:
                eval_voices = False
            pass

            if os.path.exists(reference_txt_file):
                os.remove(reference_txt_file)
            if os.path.exists(predicted_txt_file):
                os.remove(predicted_txt_file)

        print(" ----- n_voices ----- ", n_voices)
        global_res_dict.__multi_pitch__ /= n_voices
        global_res_dict.__voice__ /= n_voices
        global_res_dict.__meter__ /= n_voices
        global_res_dict.__harmony__ /= n_voices
        global_res_dict.__note_value__ /= n_voices

        return global_res_dict

    """Sequence to kern"""

    def seq2kern(sequence, name_out):
        with open(name_out, "w") as fout:
            n_cols = (np.where(np.array(sequence) == COR_TOKEN)[0][0] + 1) // 2

            # Kern header:
            fout.write("\t".join(["**kern"] * n_cols) + "\n")

            # Iterating through the line:
            line = list()
            flag_CON_TOKEN = False
            for token in sequence:
                if token == COR_TOKEN:
                    if len(line) > 0:
                        if len(line) < n_cols:
                            line.extend(["."] * (n_cols - len(line)))
                        fout.write("\t".join(line) + "\n")
                    line = list()
                elif token == COC_TOKEN:
                    pass
                elif token == CON_TOKEN:
                    flag_CON_TOKEN = True
                else:
                    if token != "DOT":
                        if flag_CON_TOKEN == True:
                            if len(line) > 0:
                                line[-1] = line[-1] + " " + token
                            else:
                                line.append(token)
                            flag_CON_TOKEN = False
                        else:
                            line.append(token)
                        pass
                    else:
                        line.append(".")
                    pass
                pass
            pass
        return

    MV2H_score = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
    ### GROUND TRUTH:
    # Creating GT kern file:
    seq2kern(sequence=y_true, name_out="gtKern.krn")

    ### PREDICTION:
    # Creating predicted kern file:
    seq2kern(sequence=y_pred, name_out="predKern.krn")

    # Testing whether predicted Kern can be processed as polyphonic
    flag_polyphonic_kern = False
    try:
        a = converterm21.parse("predKern.krn").write("midi")
    except:
        flag_polyphonic_kern = False

    if flag_polyphonic_kern:  # If predicted can be polyphonic -> Polyphonic evaluation
        res_dict = eval_as_polyphonic()
    else:  # Otherwise -> Single-voice evaluation
        res_dict = eval_as_monophonic()

    MV2H_score.__multi_pitch__ += res_dict.multi_pitch
    MV2H_score.__voice__ += res_dict.voice
    MV2H_score.__meter__ += res_dict.meter
    MV2H_score.__harmony__ += res_dict.harmony
    MV2H_score.__note_value__ += res_dict.note_value

    mv2h_dict = {
        "multi-pitch": MV2H_score.__multi_pitch__,
        "voice": MV2H_score.__voice__,
        "meter": MV2H_score.__meter__,
        "harmony": MV2H_score.__harmony__,
        "note_value": MV2H_score.__note_value__,
        "mv2h": MV2H_score.mv2h,
    }

    if os.path.exists("gtKern.krn"):
        os.remove("gtKern.krn")
    if os.path.exists("predKern.krn"):
        os.remove("predKern.krn")

    return mv2h_dict


if __name__ == "__main__":
    for idx in range(44):
        with open("utils/gt/{}.krn".format(idx)) as f:
            gt = f.readlines()[0].strip().split("\t")

        with open("utils/hyp/{}.krn".format(idx)) as f:
            hyp = f.readlines()[0].strip().split("\t")

        print(
            "{}.krn = {}".format(
                idx, compute_metrics(y_true=gt, y_pred=hyp, compute_mv2h=True)
            )
        )

    # idx = 1
    # with open("utils/gt/{}.krn".format(idx)) as f:
    #     gt = f.readlines()[0].strip().split("\t")
    # with open("utils/hyp/{}.krn".format(idx)) as f:
    #     hyp = f.readlines()[0].strip().split("\t")
    # print("{}.krn = {}".format(idx, compute_metrics(y_true = gt, y_pred = hyp, compute_mv2h = True)))

    print("end")
