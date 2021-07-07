# Music generator using LSTM
Integrate between music melody files (midi) and musix text to train an RNN-LSTM model to generate lyriics based on an initial word and melody.
MIDI files) and contain various types of information – notes, the instruments used etc.
words are represented using word2vec embedding.

Architecture:
We build two submodels for text and melody and after running them through an initial lstm layer we concatenated them and proceed to add additional dense layers before outputting a new word.
For loss and optimizer we used CategoricalCrossentropy and Adam respectively.
Also we looked at accuracy as a metric.
Pseudo code:
# submodel 1 - text
    input_1 =Input (shape=max_sequence_length)
    text_embedding_layer = Embedding()
    text_LSTM_Layer = LSTM(100)(text_embedding_layer)
# submodel 2
    input_2 = Input(shape=(PITCH_DIM, MAX_MELODY_SEQUENCE_LENGTH))
    melody_LSTM_Layer = LSTM(100)(input_2)
# concatenate
    concat_layer = Concatenate()([text_LSTM_Layer, melody_LSTM_Layer])
# rest of the model
    dense_layer = Dense(embedding_matrix.shape[1], activation='relu')(concat_layer)
    output = Dense(embedding_matrix.shape[0], activation='softmax')(dense_layer)
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    model.compile(loss= CategoricalCrossentropy(),     optimizer= Adam(), metrics=['acc'])

Integrating melody approach 1:
In the first approach, we divided the text to sequences in a fixed length 20. 
We have also sampled the “piano roll” (sum of pitches from all instruments) of the whole melody in a frequency of 1 (each second). 
Each input to the model was a pair of a text sequence and the whole melody of the song.

Integrating melody approach 2:
In the second approach, we also divided the text to sequences of 20. 
However this time we align the texts and the melodies. Each text sequence will be paired with a different, matching part of the melody.
In order to align the texts melodies we calculated for each text sequence its relative ‘location’ in its song, and cut that same relative part from the sampled melody pitches.
This approach also allowed us to sample the melody in a higher frequency. For this method we have used FS of 10 (sampling every 0.1 second). 


