import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
with open('TEXTO/xtokenizer.pickle', 'rb') as handle:
    x_tokenizer = pickle.load(handle)
with open('TEXTO/ytokenizer.pickle', 'rb') as handle:
    y_tokenizer = pickle.load(handle)

reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index


def decode_sequence(encoder_model,decoder_model,input_seq):
    input_seq    =   pad_sequences(x_tokenizer.texts_to_sequences([input_seq]) ,  maxlen=350, padding='post')
    input_seq = input_seq[0].reshape(1,350)
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index+1]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (99)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

