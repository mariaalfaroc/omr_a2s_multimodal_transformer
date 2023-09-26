import os, shutil

from pyMV2H.utils.mv2h import MV2H
from pyMV2H.utils.music import Music
from pyMV2H.metrics.mv2h import mv2h
from music21 import converter as converterm21
from pyMV2H.converter.midi_converter import MidiConverter as Converter

from vocab.preprocessing import COC_TOKEN, COR_TOKEN

# TODO:
# - Comprobar el número de voces en GrandStaff.
# - Ahora mismo se introduce el token de cambio de voz (COR_TOKEN) en la codificación,
#   pero no está introducido en el cálculo de las métricas. Comprobar que el de COC_TOKEN
#   se tiene en cuenta también.
# - Las métricas solo están pensadas para kern y no para las dos opciones de decoupl.


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
        'sym-er': 100. * ed_acc / length_acc,
        'seq-er': 100. * label_acc / len(y_pred)
    }


def compute_mv2h_metrics(y_true, y_pred):

    def krn2midi(in_file):
        a = converterm21.parse(in_file).write('midi')
        midi_file = a.name
        shutil.copyfile(a, midi_file)
        os.remove(in_file)
        return midi_file

    def midi2txt(midi_file):
        txt_file = midi_file.replace('mid', 'txt')
        converter = Converter(file=midi_file, output=txt_file)
        converter.convert_file()
        with open(txt_file,'r') as fin:
            f = [u.replace('.0', '') for u in fin.readlines()]
        with open(txt_file,'w') as fout:
            for u in f: fout.write(u)
        os.remove(midi_file)
        return txt_file


    """Polyphonic evaluation."""
    def eval_as_polyphonic():
        # Processing GT:
        reference_midi_file = krn2midi('gtKern.krn')

        # Processing prediction:
        predicted_midi_file = krn2midi('predKern.krn')

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
        # Opening file:
        with open(in_file) as fin:
            read_file = fin.readlines()

        # Read voice:
        voice = [u.split('\t')[voice].strip() for u in read_file]

        # Writing voice:
        with open(out_file, 'w') as fout:
            for token in voice:
                fout.write(token + '\n')

        return

    """Monophonic evaluation."""
    def eval_as_monophonic():
        global_res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
        # TODO
        # Check whether it is 4 or 2!
        for it_voice in range(4):
            # Processing GT:
            divide_voice('gtKern.krn', 'gtVoiceKern.krn', it_voice)
            reference_midi_file = krn2midi('gtVoiceKern.krn')

            # Processing prediction:
            divide_voice('predKern.krn', 'predVoiceKern.krn', it_voice)
            predicted_midi_file = krn2midi('predVoiceKern.krn')

            # Converting to TXT:
            ### True:
            reference_txt_file = midi2txt(reference_midi_file)

            ### Pred:
            predicted_txt_file = midi2txt(predicted_midi_file)
            
            # Figures of merit:
            reference_file = Music.from_file(reference_txt_file)
            transcription_file = Music.from_file(predicted_txt_file)

            res_dict = MV2H(multi_pitch = 0, voice = 0, meter = 0, harmony = 0, note_value = 0)
            try:
                res_dict = mv2h(reference_file, transcription_file)
                global_res_dict.__multi_pitch__  += res_dict.multi_pitch
                global_res_dict.__voice__ += res_dict.voice
                global_res_dict.__meter__ += res_dict.meter
                global_res_dict.__harmony__ += res_dict.harmony
                global_res_dict.__note_value__ += res_dict.note_value
            except:
                pass

            os.remove(reference_txt_file)
            os.remove(predicted_txt_file)

        global_res_dict.__multi_pitch__  /= 4
        global_res_dict.__voice__ /= 4
        global_res_dict.__meter__ /= 4
        global_res_dict.__harmony__ /= 4
        global_res_dict.__note_value__ /= 4

        return global_res_dict


    MV2H_global = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
    for it_piece in range(len(y_true)):
        ### GROUND TRUTH:
        # Creating GT kern file:
        with open('gtKern.krn', 'w') as fout:
            # Kern header:
            fout.write('\t'.join(['**kern']*4) + '\n')

            # Iterating through the line:
            line = []
            for token in y_true[it_piece]:
                # TODO
                # Check whether we also use the COC_TOKEN
                if token == COR_TOKEN:
                    if len(line) > 0:
                        if len(line) < 4: line.extend(['.']*(4-len(line)))
                        fout.write('\t'.join(line) + '\n')
                    line = []
                else:
                    if token != 'DOT': line.append(token)
                    else: line.append('.')

        ### PREDICTION:
        # Creating predicted kern file:
        with open('predKern.krn', 'w') as fout:
            # Kern header:
            fout.write('\t'.join(['**kern']*4) + '\n')

            # Iterating through the line:
            line = []
            for token in y_pred[it_piece]:
                # TODO
                # Check whether we also use the COC_TOKEN
                if token == COR_TOKEN:
                    if len(line) > 0:
                        if len(line) < 4: line.extend(['.']*(4-len(line)))
                        fout.write('\t'.join(line) + '\n')
                    line = []
                else:
                    if token != 'DOT': line.append(token)
                    else: line.append('.')

        # Testing whether predicted Kern can be processed as polyphonic
        flag_polyphonic_kern = True
        try:
            a = converterm21.parse('predKern.krn').write('midi')
        except:
            flag_polyphonic_kern = False

        if flag_polyphonic_kern: # If predicted can be polyphonic -> Polyphonic evaluation
            res_dict = eval_as_polyphonic()
        else: # Otherwise -> Single-voice evaluation
            res_dict = eval_as_monophonic()

        MV2H_global.__multi_pitch__  += res_dict.multi_pitch
        MV2H_global.__voice__ += res_dict.voice
        MV2H_global.__meter__ += res_dict.meter
        MV2H_global.__harmony__ += res_dict.harmony
        MV2H_global.__note_value__ += res_dict.note_value

    MV2H_global.__multi_pitch__  /= len(y_true)
    MV2H_global.__voice__ /= len(y_true)
    MV2H_global.__meter__ /= len(y_true)
    MV2H_global.__harmony__ /= len(y_true)
    MV2H_global.__note_value__ /= len(y_true)

    mv2h_dict = {
        'multi-pitch': MV2H_global.__multi_pitch__ ,
        'voice': MV2H_global.__voice__,
        'meter': MV2H_global.__meter__,
        'harmony': MV2H_global.__harmony__,
        'note_value': MV2H_global.__note_value__,
        'mv2h': MV2H_global.mv2h
    }

    return mv2h_dict
