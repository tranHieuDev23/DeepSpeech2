from process_data import id_to_char_sequence, mfcc_from_file
from model import deepspeech, get_trainable_model, get_predictable_model
import sys
import numpy as np


def compress(text_sequence):
    result = []
    last_c = None
    for c in text_sequence:
        if (c != last_c):
            result.append(c)
        last_c = c
    result = [item for item in result if item != '`']
    return ''.join(result)


def predict_with_language_model(model, mfcc):
    ml_pred = model.predict(np.array([mfcc]))[0]
    ids = np.argmax(ml_pred, axis=1)
    best_chars = id_to_char_sequence(ids)
    best_text = compress(best_chars)
    return best_text


weights_path = sys.argv[1]

n_mfcc = 128
model = get_trainable_model(deepspeech(
    is_gpu=False, feature_cnt=n_mfcc, units=256))
model = get_predictable_model(model)
model.load_weights(weights_path)

while(True):
    audio_path = input('Please input audio file path: ')
    mfcc = mfcc_from_file(audio_path, 16000, n_mfcc)
    text = predict_with_language_model(model, mfcc)
    print(text)
