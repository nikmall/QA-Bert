import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations
from transformers import TFBertModel

from utils.evaluation import calculate_scores


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

    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    sequence_emb = bert_encoder.bert(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]  # get only sequence output, throw the classifcation pooled output

    ## QA Head sub-Model
    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(sequence_emb)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(sequence_emb)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(activations.softmax, name="start_probs")(start_logits)
    end_probs = layers.Activation(activations.softmax, name="end_probs")(end_logits)

    model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    return model


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


def model_train(model, conf, epochs, x_train, y_train, x_dev, y_dev, train_squad_qas, val_squad_qas):
    exact_match_callback_train = Evaluation(x_train, y_train, train_squad_qas, "train")
    exact_match_callback_val = Evaluation(x_dev, y_dev, val_squad_qas, "validation")
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        verbose=1,
        batch_size=conf["batch"],
        callbacks=[exact_match_callback_train, exact_match_callback_val],
    )
    max_len = conf["max_len"]
    model.save(f"model_{max_len}")


class Evaluation(tf.keras.callbacks.Callback):
    """
    This evaluation callback calculates the EM and F1 score after each epoch.
    Each SquadQuestionAnswer object has the offsets from the Bert tokenizer. The
    offset gives character level offsets in the raw text context .
    Using the offsets we find the span of text corresponding to the tokens between the
    predicted start and end indexes, getting the string of predicted answer. Finally,
    calculate  of EM and F1 score of each predicted question compared to all possible
    answers keeping the max score for each Question.  Print the averaged final scores.
    """

    def __init__(self, x_val, y_val, val_squad_qas, message=''):
        self.x_val = x_val
        self.y_val = y_val
        self.val_squad_qas = val_squad_qas
        self.message = message

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_val)
        em = []
        f1 = []

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

            exact_score, f1_score = calculate_scores(qa_i.all_answers, pred_answer)
            em.append(exact_score)
            f1.append(f1_score)
        em_total = sum(em) / len(self.y_val[0])
        f1_total = sum(f1) / len(self.y_val[0])
        print(f"\nOn epoch: {epoch + 1}, Exact Match score: {em_total:.2f} "
              f"and F1 score: {f1_total:.2f} on {self.message} data.")
