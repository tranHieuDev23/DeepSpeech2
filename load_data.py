import numpy as np
from multiprocessing import Pool, cpu_count
from process_data import line_to_mfcc_and_transcript, line_to_transcript, alphabet
from os import listdir
from os.path import isfile, join, split
from nltk.util import pad_sequence
from tqdm import tqdm


def load_transcript_filenames(root_folder):
    result = []
    for entry in listdir(root_folder):
        full_path = join(root_folder, entry)
        if (isfile(full_path)):
            if (entry.endswith('trans.txt')):
                result.append(full_path)
        else:
            result.extend(load_transcript_filenames(full_path))
    return result


def load_data_from_folder(root_folder, sample_rate, n_mfcc, dataset_size):
    all_transcript = load_transcript_filenames(root_folder)
    X = []
    X_length = []
    y = []
    y_length = []
    item_count = 0
    pool = Pool(cpu_count() - 1)
    for t in tqdm(all_transcript):
        audio_directory = split(t)[0]
        transcript_file = open(t)
        lines = transcript_file.readlines()
        inputs = [(audio_directory, line, sample_rate, n_mfcc)
                  for line in lines]
        for processed_line in pool.imap_unordered(line_to_mfcc_and_transcript, inputs, 16):
            (audio_mfcc, id_vector) = processed_line
            X.append(audio_mfcc)
            X_length.append(audio_mfcc.shape[0])
            y.append(np.asarray(id_vector))
            y_length.append(len(id_vector))
            item_count += 1
            if (item_count >= dataset_size):
                break
        transcript_file.close()
        if (item_count >= dataset_size):
            break
    pool.close()
    pool.join()
    return X, X_length, y, y_length, item_count


class DatasetGenerator:
    def __init__(self, X, X_length, y, y_length, item_count, batch_size=64, n_mfcc=128):
        self.X = X
        self.X_length = X_length
        self.y = y
        self.y_length = y_length
        self.item_count = item_count
        self.batch_size = batch_size
        self.n_mfcc = n_mfcc

    def __len__(self):
        return (self.item_count + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        from_id = index * self.batch_size
        to_id = min(from_id + self.batch_size, self.item_count)
        item_count = to_id - from_id
        X_max_length = np.max(self.X_length[from_id:to_id])
        y_max_length = np.max(self.y_length[from_id:to_id])
        batch_X = np.zeros((item_count, X_max_length, self.n_mfcc))
        batch_y = np.zeros((item_count, y_max_length))
        for i in range(from_id, to_id):
            X_i = self.X[i]
            y_i = self.y[i]
            batch_X[i - from_id, :X_i.shape[0], :X_i.shape[1]] = X_i
            batch_y[i - from_id, :y_i.shape[0]] = y_i
        batch_X_length = np.array(self.X_length[from_id:to_id])
        batch_y_length = np.array(self.y_length[from_id:to_id])
        return [batch_X, batch_y, batch_X_length, batch_y_length], batch_y


def load_transcript_corpus(root_folder):
    all_transcript = load_transcript_filenames(root_folder)
    tokenized_transcripts = []
    pool = Pool(cpu_count() - 1)
    for t in tqdm(all_transcript):
        transcript_file = open(t)
        for result in pool.imap_unordered(line_to_transcript, transcript_file, 16):
            tokenized_transcripts.append(result)
        transcript_file.close()
    return tokenized_transcripts
