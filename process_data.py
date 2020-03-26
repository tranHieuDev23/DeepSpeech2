import numpy as np
import librosa
from os.path import join


def mfcc_from_file(filepath, sample_rate, n_mfcc):
    samples, sr = librosa.load(filepath, sample_rate)
    mfcc = librosa.feature.mfcc(samples, sr, n_mfcc=n_mfcc)
    return mfcc.transpose()


alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ', '\'', '`']
char_to_id = {}
id_to_char = {}
id = 0
for item in alphabet:
    char_to_id[item] = id
    id_to_char[id] = item
    id += 1


def text_to_id_sequence(text):
    return [char_to_id[c] for c in text]


def id_to_char_sequence(ids):
    return [id_to_char[i] for i in ids]


def line_to_mfcc_and_transcript(args):
    (audio_directory, line, sample_rate, n_mfcc) = args
    first_space = line.find(' ')
    # Load and preprocess the audio
    audio_filename = line[:first_space] + ".flac"
    audio_path = join(audio_directory, audio_filename)
    audio_mfcc = mfcc_from_file(audio_path, sample_rate, n_mfcc)
    # Load and preprocess the transcript
    transcript = line[first_space + 1:].strip()
    id_vector = text_to_id_sequence(transcript)
    return (audio_mfcc, id_vector)


def line_to_transcript(line):
    first_space = line.find(' ')
    transcript = line[first_space + 1:].strip()
    characters = [c for c in transcript]
    return characters
