
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import logging
import numpy as np
import dill as dpickle


def load_text_processor(fname='title_pp.dpkl'):
    # Load files from disk
    with open(fname, 'rb') as f:
        pp = dpickle.load(f)

    num_tokens = max(pp.id2token.keys()) + 1
    print(f'Size of vocabulary for {fname}: {num_tokens:,}')
    return num_tokens, pp


def load_decoder_inputs(decoder_np_vecs='train_title_vecs.npy'):
    vectorized_title = np.load(decoder_np_vecs)
    decoder_input_data = vectorized_title[:, :-1]
    decoder_target_data = vectorized_title[:, 1:]

    print(f'Shape of decoder input: {decoder_input_data.shape}')
    print(f'Shape of decoder target: {decoder_target_data.shape}')
    return decoder_input_data, decoder_target_data


def load_encoder_inputs(encoder_np_vecs='train_body_vecs.npy'):
    vectorized_body = np.load(encoder_np_vecs)
    encoder_input_data = vectorized_body
    doc_length = encoder_input_data.shape[1]
    print(f'Shape of encoder input: {encoder_input_data.shape}')
    return encoder_input_data, doc_length


def viz_model_architecture(model):
    """Visualize model architecture in Jupyter notebook."""
    display(SVG(model_to_dot(model).create(prog='dot', format='svg')))


def extract_encoder_model(model):
    encoder_model = model.get_layer('Encoder-Model')
    return encoder_model


def extract_decoder_model(model):
    latent_dim = model.get_layer('Decoder-Word-Embedding').output_shape[-1]
    decoder_inputs = model.get_layer('Decoder-Input').input
    dec_emb = model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
    dec_bn = model.get_layer('Decoder-Batchnorm-1')(dec_emb)

    gru_inference_state_input = Input(
        shape=(latent_dim,), name='hidden_state_input')
    gru_out, gru_state_out = model.get_layer(
        'Decoder-GRU')([dec_bn, gru_inference_state_input])

    dec_bn2 = model.get_layer('Decoder-Batchnorm-2')(gru_out)
    dense_out = model.get_layer('Final-Output-Dense')(dec_bn2)
    decoder_model = Model([decoder_inputs, gru_inference_state_input],
                          [dense_out, gru_state_out])
    return decoder_model


class Seq2Seq_Inference(object):
    def __init__(self,
                 encoder_preprocessor,
                 decoder_preprocessor,
                 seq2seq_model):

        self.pp_body = encoder_preprocessor
        self.pp_title = decoder_preprocessor
        self.seq2seq_model = seq2seq_model
        self.encoder_model = extract_encoder_model(seq2seq_model)
        self.decoder_model = extract_decoder_model(seq2seq_model)
        self.default_max_len_title = self.pp_title.padding_maxlen
        self.nn = None
        self.rec_df = None

    def generate_issue_title(self,
                             raw_input_text,
                             max_len_title=None):
        if max_len_title is None:
            max_len_title = self.default_max_len_title
        raw_tokenized = self.pp_body.transform([raw_input_text])
        body_encoding = self.encoder_model.predict(raw_tokenized)
        original_body_encoding = body_encoding
        state_value = np.array(self.pp_title.token2id['_start_']).reshape(1, 1)
        decoded_sentence = []
        stop_condition = False
        while not stop_condition:
            preds, st = self.decoder_model.predict(
                [state_value, body_encoding])

            pred_idx = np.argmax(preds[:, :, 2:]) + 2
            pred_word_str = self.pp_title.id2token[pred_idx]

            if pred_word_str == '_end_' or len(decoded_sentence) >= max_len_title:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)
            body_encoding = st
            state_value = np.array(pred_idx).reshape(1, 1)

        return original_body_encoding, ' '.join(decoded_sentence)

    def print_example(self,
                      i,
                      body_text,
                      title_text,):
        if i:
            print(f'Example # {i}')
        print(f"Issue Body:\n {body_text} \n")
        if title_text:
            print(f"Original Title: {title_text}")
        emb, gen_title = self.generate_issue_title(body_text)
        print(
            f"\n****** Machine Generated Title (Prediction) ******: {gen_title}\n")

    def demo_model_predictions(self,
                               n,
                               issue_df):
        body_text = issue_df.body.tolist()
        title_text = issue_df.issue_title.tolist()

        demo_list = np.random.randint(low=1, high=len(body_text), size=n)
        for i in demo_list:
            self.print_example(i,
                               body_text=body_text[i],
                               title_text=title_text[i])
