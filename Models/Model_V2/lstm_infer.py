import pandas as pd
import logging
import glob
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 500)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
traindf, testdf = train_test_split(pd.read_csv('github_issues.csv').sample(n=200000),
                                   test_size=.10)

train_body_raw = traindf.body.tolist()
train_title_raw = traindf.issue_title.tolist()

latent_dim = 300
from ktext.preprocess import processor

body_pp = processor(keep_n=8000, padding_maxlen=70)
train_body_vecs = body_pp.fit_transform(train_body_raw)

title_pp = processor(append_indicators=True, keep_n=4500,
                     padding_maxlen=12, padding ='post')

# process the title data
train_title_vecs = title_pp.fit_transform(train_title_raw)

import dill as dpickle
import numpy as np

# Save the preprocessor
with open('body_pp.dpkl', 'wb') as f:
    dpickle.dump(body_pp, f)

with open('title_pp.dpkl', 'wb') as f:
    dpickle.dump(title_pp, f)

# Save the processed data
np.save('train_title_vecs.npy', train_title_vecs)
np.save('train_body_vecs.npy', train_body_vecs)

from seq2seq_utils import load_decoder_inputs, load_encoder_inputs, load_text_processor

encoder_input_data, doc_length = load_encoder_inputs('train_body_vecs.npy')
decoder_input_data, decoder_target_data = load_decoder_inputs('train_title_vecs.npy')
sum_length = decoder_input_data.shape[1]

num_encoder_tokens, body_pp = load_text_processor('body_pp.dpkl')
num_decoder_tokens, title_pp = load_text_processor('title_pp.dpkl')

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Dropout, Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization, SimpleRNN, RepeatVector, Flatten, TimeDistributed
from tensorflow.keras import optimizers


def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(n_input,))
    x = Embedding(num_encoder_tokens, latent_dim, name='Body-Word-Embedding', mask_zero=False)(encoder_inputs)
    x = BatchNormalization(name='Encoder-Batchnorm-1')(x)
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(x)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)

    # define training decoder
    decoder_inputs = Input(shape=(None,))
    dec_emb = Embedding(num_decoder_tokens, latent_dim, name='Decoder-Word-Embedding', mask_zero=False)(decoder_inputs)
    dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_bn, initial_state=encoder_states)
    x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_outputs)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(x)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference decoder
    decoder_state_input_h = Input(shape=(None,))
    decoder_state_input_c = Input(shape=(None,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(dec_bn, initial_state=decoder_states_inputs)
    x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_outputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(x)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

script_name_base = 'tutorial_seq2seq'
csv_logger = CSVLogger('{:}.log'.format(script_name_base))
model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                   save_best_only=True)

batch_size = 1200
epochs = 1

model, encoder_model, decoder_model = define_models(doc_length, 11, 300)
model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='mse')
model.summary()

decoder_model.summary()
history = model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1),
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.12, callbacks=[csv_logger, model_checkpoint])

#save model
model.save('model_tutorial.h5')


class Seq2Seq_Inference(object):
    def __init__(self,
                 encoder_preprocessor,
                 decoder_preprocessor,):

        self.pp_body = encoder_preprocessor
        self.pp_title = decoder_preprocessor
        self.seq2seq_model, self.encoder_model, self.decoder_model = model, encoder_model, decoder_model
        self.default_max_len_title = self.pp_title.padding_maxlen
        self.nn = None
        self.rec_df = None

    def generate_issue_title(self,
                             raw_input_text,
                             max_len_title=None):
        """
        Use the seq2seq model to generate a title given the body of an issue.
        Inputs
        ------
        raw_input: str
            The body of the issue text as an input string
        max_len_title: int (optional)
            The maximum length of the title the model will generate
        """
        if max_len_title is None:
            max_len_title = self.default_max_len_title
        # get the encoder's features for the decoder
        raw_tokenized = self.pp_body.transform([raw_input_text])
        body_encoding = self.encoder_model.predict(raw_tokenized)
        # we want to save the encoder's embedding before its updated by decoder
        #   because we can use that as an embedding for other tasks.
        original_body_encoding = body_encoding
        state_value = np.array(self.pp_title.token2id['_start_']).reshape(1, 1)
        # state_value = np.array([0.0 for _ in range(70)]).reshape(1, 1, 70)
        decoded_sentence = []
        stop_condition = False
        while not stop_condition:
            preds, st_h, st_c = self.decoder_model.predict([state_value] + body_encoding)
            # We are going to ignore indices 0 (padding) and indices 1 (unknown)
            # Argmax will return the integer index corresponding to the
            #  prediction + 2 b/c we chopped off first two
            pred_idx = np.argmax(preds[:, :, 2:]) + 2

            # retrieve word from index prediction
            pred_word_str = self.pp_title.id2token[pred_idx]

            if pred_word_str == '_end_' or len(decoded_sentence) >= max_len_title:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)

            # update the decoder for the next word
            body_encoding = [st_h, st_c]
            state_value = np.array(pred_idx).reshape(1, 1)

        return original_body_encoding, ' '.join(decoded_sentence)

    def print_example(self,
                      i,
                      body_text,
                      title_text,
                      threshold):


        emb, gen_title = self.generate_issue_title(body_text)
        # print(f"\n****** Machine Generated Title (Prediction) ******:\n {gen_title}")
        print(gen_title)
        if self.nn:
            # return neighbors and distances
            n, d = self.nn.get_nns_by_vector(emb.flatten(), n=4,
                                             include_distances=True)
            neighbors = n[1:]
            dist = d[1:]

            if min(dist) <= threshold:
                cols = [ 'issue_title', 'body']
                dfcopy = self.rec_df.iloc[neighbors][cols].copy(deep=True)
                dfcopy['dist'] = dist
                similar_issues_df = dfcopy.query(f'dist <= {threshold}')

                print("\n**** Similar Issues (using encoder embedding) ****:\n")
                display(similar_issues_df)

        return gen_title

    def demo_model_predictions(self,
                               n,
                               issue_df,
                               threshold=1):
        """
        Pick n random Issues and display predictions.
        Input:
        ------
        n : int
            Number of issues to display from issue_df
        issue_df : pandas DataFrame
            DataFrame that contains two columns: `body` and `issue_title`.
        threshold : float
            distance threshold for recommendation of similar issues.
        Returns:
        --------
        None
            Prints the original issue body and the model's prediction.
        """
        # Extract body and title from DF
        body_text = issue_df.body.tolist()
        title_text = issue_df.issue_title.tolist()

        demo_list = np.random.randint(low=1, high=len(body_text), size=n)
        gen_title = []
        for i in demo_list:
            gen_title.append(self.print_example(i,
                                                body_text=body_text[i],
                                                title_text=title_text[i],
                                                threshold=threshold))

        return body_text, title_text, gen_title


seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                 decoder_preprocessor=title_pp)
# this method displays the predictions on random rows of the holdout set
seq2seq_inf.demo_model_predictions(n=5, issue_df=testdf)

from rouge import Rouge 

rouge = Rouge()

test_title_text = testdf.issue_title.tolist()
test_body_text = testdf.body.tolist()
predict_title_text = [None]*len(test_body_text)
print(predict_title_text)
rouge_1_p, rouge_1_r, rouge_1_f, rouge_2_p, rouge_2_r, rouge_2_f, rouge_l_p, rouge_l_f, rouge_l_r = 0, 0, 0, 0, 0, 0, 0, 0, 0
for i in range(len(test_body_text)):
    exm, predict_title_text[i] = seq2seq_inf.generate_issue_title(raw_input_text = test_body_text[i])
    scores = rouge.get_scores(predict_title_text[i], test_title_text[i])
    rouge_1_f = rouge_1_f + scores[0]['rouge-1']['f']

    rouge_2_f = rouge_2_f + scores[0]['rouge-2']['f']

    rouge_l_f = rouge_l_f + scores[0]['rouge-l']['f']

print("ROUGE-1:", rouge_1_f/len(test_body_text))
print("ROUGE-2:",rouge_2_f/len(test_body_text))
print("ROUGE-l:",rouge_l_f/len(test_body_text))
print("Average of ROUGE-1, ROUGE-2 and ROUGE-l: ", (rouge_1_f + rouge_2_f + rouge_l_f)/3/len(test_body_text))

