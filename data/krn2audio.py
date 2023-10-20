import os, sys
from pathlib import Path
import multiprocessing as mp

from music21 import *
from midi2audio import FluidSynth

SOUND_FONT = 'data/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2'

def krn2audio(krn_file):
	filename = os.path.split(krn_file)[0] + '/' + os.path.split(krn_file)[1].split('.')[0]
	if os.path.split(krn_file)[1][:1] == '.':
		print('Skipped')
		return filename + '\t skipped'
	
	# krn to midi
	krn_stream = 0
	try:
		krn_stream = converter.parse(krn_file)
	except Exception as err:
		print(f'File {krn_file} raised unexpected error: {type(err)}, {err}')
		return filename + '\t ERR'

	midi_file = filename + '.mid'
	midi_stream = krn_stream.write('midi', fp=midi_file)
	
	# midi to wav
	wav_file = filename + '.wav'
	fs = FluidSynth(sample_rate=22050, sound_font=SOUND_FONT)
	fs.midi_to_audio(midi_file, wav_file)

	os.remove(midi_file)
	
	return filename + '\t OK'


def main():
	krn_files = Path('.').rglob('*/*.krn')
	pool = mp.Pool(processes=mp.cpu_count())
	results = pool.map(krn2audio, krn_files)
	pool.join()
	pool.close()
	
	print(results, file=sys.stderr)

if __name__ == '__main__':
	main()
