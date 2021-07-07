import itertools
import tensorflow as tf
import pandas as pd
import pretty_midi
import numpy as np
import os
import gensim
import math
import pathlib
import copy

BASE_PATH = '/home/odedrot/DL_ex_3'
TEST_SET_PATH = BASE_PATH+'/data/lyrics_test_set.csv'
TRAIN_SET_PATH = BASE_PATH+'/data/lyrics_train_set.csv'
MIDI_FILES_PATH = BASE_PATH+'/data/midi_files'
WORD2VEC_MODEL_PATH = BASE_PATH+'/glove.6B.300d.txt'

MAX_TEXT_SEQUENCE_LENGTH = 20
MAX_MELODY_SEQUENCE_LENGTH = 500
PAD_INT_VALUE = 0
PAD_WORD = '<PAD>'

PIANO_ROLL_FS = 1
PITCH_DIM = 128


def load_data(path):
    df = pd.DataFrame(columns=['artist', 'song_name', 'lyrics'])

    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        artist, song_name, lyrics = line.split(',', maxsplit=2)

        # Cut the &,,,, which indicates a song ending
        lyrics = lyrics[:-7]

        df = df.append({
            'artist': artist,
            'song_name': song_name,
            'lyrics': lyrics
        }, ignore_index=True)

    return df


class SongMetadata:
    def __init__(self, artist, song_name, song_len):
        self.artist = artist
        self.song_name = song_name
        self.song_len = song_len
        self.relative_place = None

    def __repr__(self):
        return SongMetadata({self.artist}, {self.song_name}, {self.song_len}, {self.relative_place})


def extract_texts(df, seq_len):
    texts = []
    index_to_metadata = {}

    # Generate text sequences in the requested length +1
    for song in df.itertuples():
        # Ignore the new-line character
        song_lyrics = song.lyrics.replace('& ', '')
        song_lyrics = song_lyrics.split()
        song_metadata = SongMetadata(song.artist, song.song_name.strip(), len(song_lyrics))

        for i in range(len(song_lyrics) - seq_len + 1):
            # Set the metadata according to the relative place of the text in the song
            curr_text_metadata = copy.copy(song_metadata)
            curr_text_metadata.relative_place = i / (len(song_lyrics) - seq_len)
            index_to_metadata[len(texts)] = curr_text_metadata

            if i + seq_len + 1 >= len(song_lyrics):
                texts.append(' '.join(song_lyrics[i:]))
            else:
                texts.append(' '.join(song_lyrics[i:i + seq_len + 1]))

    return texts, index_to_metadata


def process_data(tr_df, val_df, seq_len):
    # Extract texts
    tr_texts, tr_metadata = extract_texts(tr_df, seq_len)
    val_texts, val_metadata = extract_texts(val_df, seq_len)

    # Tokenize the text - both train and validation so the embedding matrix will be complete.
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(itertools.chain(tr_texts, val_texts))
    tokenizer.word_index[PAD_WORD] = PAD_INT_VALUE
    tokenizer.index_word[PAD_INT_VALUE] = PAD_WORD

    # Encode the words in the text to integers (the index in the tokenizer)
    tr_texts = tokenizer.texts_to_sequences(tr_texts)
    tr_texts = tf.keras.preprocessing.sequence.pad_sequences(tr_texts, maxlen=seq_len + 1, truncating='pre',
                                                             value=PAD_INT_VALUE)
    val_texts = tokenizer.texts_to_sequences(val_texts)
    val_texts = tf.keras.preprocessing.sequence.pad_sequences(val_texts, maxlen=seq_len + 1, truncating='pre',
                                                              value=PAD_INT_VALUE)

    # The first seq_len words will be the input, the last word is the label.
    X_tr = tr_texts[:, :-1]
    y_tr = tr_texts[:, -1]
    y_tr = tf.keras.utils.to_categorical(y_tr, num_classes=len(tokenizer.word_index))
    X_val = val_texts[:, :-1]
    y_val = val_texts[:, -1]
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(tokenizer.word_index))

    return X_tr, y_tr, X_val, y_val, tokenizer, tr_metadata, val_metadata


def extract_embedding_matrix(embedding_model, tokenizer): # gensim.models.keyedvectors.Word2VecKeyedVectors
    embedding_matrix = np.zeros((len(tokenizer.word_index), embedding_model.vector_size))
    for word, word_idx in tokenizer.word_index.items():
        try:
            word_vector = embedding_model.word_vec(word)
            embedding_matrix[word_idx] = word_vector
        except KeyError:
            # That means that a word is not in the vocabulary, keep it all zeros
            pass

    return embedding_matrix


def adjust_sequence_length(seq, requested_seq_len):
    # If the song is too long, cut its tail
    if seq.shape[1] > requested_seq_len:
        return seq[:, :requested_seq_len]

    # Else, pad the song with zeros
    return np.concatenate((seq, np.zeros(shape=(PITCH_DIM, requested_seq_len - seq.shape[1]))), axis=1)


def generate_lyrics(model, tokenizer, midi_file_path, first_word, n_lyrics, relative=False):
    # Load the melody data
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    piano_roll = pm.get_piano_roll(PIANO_ROLL_FS)
    if not relative:
        input_melody = adjust_sequence_length(piano_roll, MAX_MELODY_SEQUENCE_LENGTH)
        input_melody = np.array([input_melody])
        input_melodies = [input_melody] * n_lyrics
    else:
        # Each input should receive its relative part in terms of number of lyrics.
        input_melodies = []
        for i in range(n_lyrics):
            word_melody_size = piano_roll.shape[1] / n_lyrics
            text_melody_size = word_melody_size * MAX_TEXT_SEQUENCE_LENGTH
            relative_piano_roll = adjust_sequence_length(
                piano_roll[:, int(i * text_melody_size):int((i + 1) * text_melody_size)],
                MAX_MELODY_SEQUENCE_LENGTH)
            relative_piano_roll = np.array([relative_piano_roll])
            input_melodies.append(relative_piano_roll)

    text = [first_word]
    for i in range(n_lyrics):
        # Cut the last MAX_LENGTH words from the current text.
        input_text = text[-MAX_TEXT_SEQUENCE_LENGTH:]

        # Encode it as integers
        input_text = [x[0] for x in tokenizer.texts_to_sequences(input_text)]

        # Pad the input if needed
        input_text = tf.keras.preprocessing.sequence.pad_sequences([input_text],
                                                                   maxlen=MAX_TEXT_SEQUENCE_LENGTH,
                                                                   truncating='pre',
                                                                   value=PAD_INT_VALUE)

        # Predict the probabilities for the next word
        next_word_probs = model.predict([input_text, input_melodies[i]])[0]

        # Sample from the words
        next_word_index = np.random.choice(range(len(tokenizer.word_index)), p=next_word_probs)

        # Add the chosen word to the generated text
        text.append(tokenizer.index_word[next_word_index])

    return ' '.join(text)


def build_model(embedding_matrix, max_sequence_length):
    ### The new model
    # submodel 1 - text
    input_1 = tf.keras.layers.Input(shape=max_sequence_length)
    text_embedding_layer = tf.keras.layers.Embedding(output_dim=embedding_matrix.shape[1],
                                                     weights=[embedding_matrix],
                                                     input_dim=embedding_matrix.shape[0],
                                                     input_length=max_sequence_length,
                                                     trainable=False,
                                                     mask_zero=True)(input_1)
    text_LSTM_Layer = tf.keras.layers.LSTM(100)(text_embedding_layer)

    # submodel 2
    input_2 = tf.keras.layers.Input(shape=(PITCH_DIM, MAX_MELODY_SEQUENCE_LENGTH))
    melody_LSTM_Layer = tf.keras.layers.LSTM(100)(input_2)

    # concatenate
    concat_layer = tf.keras.layers.Concatenate()([text_LSTM_Layer, melody_LSTM_Layer])

    # rest of the model
    dense_layer = tf.keras.layers.Dense(embedding_matrix.shape[1], activation='relu')(concat_layer)
    output = tf.keras.layers.Dense(embedding_matrix.shape[0], activation='softmax')(dense_layer)
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

    return model


class LyricsMelodyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 train_csv_path,
                 test_csv_path,
                 mid_files_path,
                 text_seq_len,
                 melody_seq_len,
                 batch_size,
                 relative):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.mid_files_path = mid_files_path
        self.text_seq_len = text_seq_len
        self.melody_seq_len = melody_seq_len
        self.relative = relative

        # Load all of the songs in the data set
        self.tr_df = load_data(self.train_csv_path)
        self.val_df = load_data(self.test_csv_path)

        # Process the lyrics to sequences of MAX_LENGTH
        (self.X_tr,
         self.y_tr,
         self.X_val_text,
         self.y_val,
         self.tokenizer,
         self.tr_metadata,
         self.val_metadata) = process_data(self.tr_df, self.val_df, self.text_seq_len)

        self.X_val_melody = self.get_melodies(self.val_metadata, range(self.X_val_text.shape[0]))
        self.batch_size = batch_size

    def get_melodies(self, metadata, indices):
        all_metadata = set([(metadata[idx].artist, metadata[idx].song_name) for idx in indices])
        meta_to_roll = {}

        # Retrieve all of the required piano rolls
        for filename in os.listdir(self.mid_files_path):
            try:
                # Cut the tail of the file name, it is not important ('live edition', etc.)
                artist, song_name = filename.split('-')[:2]

                # Transform the artist and song name to the same format as the CSV files
                artist = artist.lower().replace('_', ' ').strip()
                song_name = pathlib.Path(song_name).stem.lower().replace('_', ' ').strip()

                # If we need this file melody, open it and extract its piano roll
                if (artist, song_name) in all_metadata:
                    pm = pretty_midi.PrettyMIDI(os.path.join(self.mid_files_path, filename))
                    piano_roll = pm.get_piano_roll(PIANO_ROLL_FS)
                    meta_to_roll[(artist, song_name)] = piano_roll

            except Exception as e:
                # We will put zeros later
                pass
                print('\n\n ERROR: {filename} got {e}\n\n')

        # For each index, return the proper piano roll for it
        ret = []
        for idx in indices:
            i = metadata[idx].relative_place
            full_piano_roll = meta_to_roll.get((metadata[idx].artist, metadata[idx].song_name),
                                               np.zeros(shape=(PITCH_DIM, MAX_MELODY_SEQUENCE_LENGTH)))
            if self.relative:
                # Each word in the text has a relative number of pitches 'attached' to it
                word_melody_size = full_piano_roll.shape[1] / metadata[idx].song_len
                text_melody_size = word_melody_size * self.text_seq_len
                relative_piano_roll = full_piano_roll[:, int(i * text_melody_size):int((i + 1) * text_melody_size)]
                ret.append(adjust_sequence_length(relative_piano_roll, self.melody_seq_len))
            else:
                ret.append(adjust_sequence_length(full_piano_roll, self.melody_seq_len))

        return np.array(ret)

    def __len__(self):
        return int(self.X_tr.shape[0] / self.batch_size)

    def __getitem__(self, index):
        """Return batch of the given index"""
        indices = range(index * self.batch_size, (index + 1) * self.batch_size)

        # Just for the last batch, there might be redundant indices
        if index == len(self) - 1:
            indices = [idx for idx in indices if idx < self.X_tr.shape[0]]

        X_text = self.X_tr[indices, :]
        X_melody = self.get_melodies(self.tr_metadata, indices)
        y = self.y_tr[indices, :]

        return [X_text, X_melody], y
