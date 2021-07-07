import argparse
from hw3 import *



def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='For only lyrics generation, load-model without epochs but with start-word')
    parser.add_argument('--train-model', help='Train a new model', required=False)
    parser.add_argument('--load-model', help='Load an existing model', required=False)
    parser.add_argument('--start-word', help='Word to start generating lyrics from', required=False)
    parser.add_argument('--n-lyrics', help='Number of lyrics to generate', required=False)
    parser.add_argument('--midi-path', help='Melody to use when generating lyrics', required=False)
    parser.add_argument('--epochs', help='Number of epochs to train', required=False)
    parser.add_argument('--relative',
                        help='Whether the input melody should be aligned to the text, '
                             'put "1" to next the relative argument (--relative 1)',
                        required=False)
    args = vars(parser.parse_args())
    
    BASE_PATH = '/home/odedrot/DL_ex_3'
    
    GENSIM_GLOVE_VECTORS_PATH = BASE_PATH + '/gensim_glove_vectors.txt'

    # Load word2vec model
    word2vec_model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(GENSIM_GLOVE_VECTORS_PATH)

    # Load and pre-process the data
    relative = False if args['relative'] is None else True
    data_gen = LyricsMelodyDataGenerator(TRAIN_SET_PATH,
                                         TEST_SET_PATH,
                                         MIDI_FILES_PATH,
                                         MAX_TEXT_SEQUENCE_LENGTH,
                                         MAX_MELODY_SEQUENCE_LENGTH,
                                         64,
                                         relative)

    # Extract an embedding matrix according to the vocabulary in the data and the word2vec model
    embedding_matrix = extract_embedding_matrix(word2vec_model, data_gen.tokenizer)

    # NOTE: you can train a new model or load an existing one, choose the proper block for you to run

    # Get the proper model
    model = None

    if args['load_model'] is not None:
        # Load an existing model
        
        model = tf.keras.models.load_model('/home/odedrot/DL_ex_3/non_relative_model_3_epochs.h5')  #BASE_PATH+args["load_model"])
        
    elif args['train_model'] is not None:
        # Build the model
        model = build_model(embedding_matrix=embedding_matrix,
                            max_sequence_length=data_gen.X_tr.shape[1])

    if model is None:
        raise ValueError("You must give a model to work with!")

    # Train the model if required
    if args['epochs'] is not None:
        history = model.fit(data_gen,
                            epochs=int(args['epochs']),
                            validation_data=([data_gen.X_val_text, data_gen.X_val_melody], data_gen.y_val),
                            verbose=1,
                            use_multiprocessing=True)
        model.save(BASE_PATH+args["train_model"]+'.h5')

    # Generate lyrics from using the trained model
    if args['start_word'] is not None and args['midi_path'] is not None and args['n_lyrics'] is not None:
        print(generate_lyrics(model,
                              data_gen.tokenizer,
                              args['midi_path'],
                              args['start_word'],
                              int(args['n_lyrics']),
                              relative))


if __name__ == '__main__':
    main()
