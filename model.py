import tensorflow as tf
from tensorflow.keras import layers, activations
from transformers import BertTokenizer, TFBertModel, BertConfig
import numpy as np
from utils.data_utils import normalize_text


def qa_bert_model(conf):
    """
    Create the Question-Answering Model using BERT encoder and a Question answering head.
    The Bert encoder is trainable and is fine tuned for outputing end and start token index
    for question answering task.
    :param conf: A dict with model development parameter
    :return:
    """
    max_len = conf["max_len"]

    # Download pre-trained model and configuration from huggingface and cache locally.
    bert_encoder = TFBertModel.from_pretrained(conf["bert_pre_trained"])

    ## QA Head sub-Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    sequence_emb = bert_encoder.bert(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0] # get only sequence output, throw the classifcation pooled output

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(sequence_emb)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(sequence_emb)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(activations.softmax)(start_logits)
    end_probs = layers.Activation(activations.softmax)(end_logits)

    model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    return model


"""
This code should preferably be run on Google Colab TPU runtime.
With Colab TPUs, each epoch will take 5-6 minutes.
"""
""""""
def build_model(conf):
    model = qa_bert_model(conf)
    model.summary()

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=conf['learning_rate'])
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model

def create_model(conf, use_tpu):
    if use_tpu:
        # Create distribution strategy
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(tpu)
        with strategy.scope():
            model = build_model(conf)
    else:
        model = build_model(conf)

    return model

"""
## Train and Evaluate
"""
def model_train(model, conf, epochs, x_train, y_train, x_dev, y_dev, val_squad_qas):
    exact_match_callback = ExactMatch(x_dev, y_dev, val_squad_qas)
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        verbose=1,
        batch_size=conf["batch"],
        callbacks=[exact_match_callback],
    )
    model.save('model')


"""
## Create evaluation Callback
This callback will compute the exact match score using the validation data
after every epoch.
"""

class ExactMatch(tf.keras.callbacks.Callback):
    """
    Each `SquadExample` object contains the character level offsets for each token
    in its input paragraph. We use them to get back the span of text corresponding
    to the tokens between our predicted start and end tokens.
    All the ground-truth answers are also present in each `SquadExample` object.
    We calculate the percentage of data points where the span of text obtained
    from model predictions matches one of the ground-truth answers.
    """

    def __init__(self, x_val, y_val, val_squad_qas):
        self.x_val = x_val
        self.y_val = y_val
        self.val_squad_qas = val_squad_qas

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_val)
        em_count = 0
        for i, (start, end) in enumerate(zip(pred_start, pred_end)):
            qa_i = self.val_squad_qas[i]
            offsets = qa_i.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_answer = qa_i.context[pred_char_start:pred_char_end]
            else:
                pred_answer = qa_i.context[pred_char_start:]

            normalized_pred_answer = normalize_text(pred_answer)
            normalized_true_answer = [normalize_text(x) for x in qa_i.all_answers]
            if normalized_pred_answer in normalized_true_answer:
                em_count += 1
        acc = em_count / len(self.y_val[0])
        print(f"\nepoch: {epoch+1}, Exact Match score: {acc:.2f}")
