import argparse
import json
import numpy as np
from tensorflow import keras

from utils.data_utils import normalize_text, create_squad_qas, create_inputs_targets
from utils.model_utils import create_bert_tokenizer, load_bert_tokenizer, set_seeds
import config
from model import create_model, model_train


def load_squad_dataset(conf, tokenizer):
    """
    A method that loads the json datfiles, creates the data SQUAD_QA objects
    and finally creates the required by Bert transformer input Ids, Tokens and
    attention mask, along with the output start and end tokens.
    :param conf: The dictionary of configuration parameters,
    :param tokenizer: The general pre-trained Bert tokenizer
    :return: x_train, y_train, x_dev, y_dev: the training and testing data
    """

    max_len = conf['max_len']

    with open(conf["train_path"]) as f:
        train_json = json.load(f)
    train_squad_qas = create_squad_qas(train_json, tokenizer, max_len)
    train_squad_qas = [x for x in train_squad_qas if x.skip is False]
    print(f"{len(train_squad_qas)} training points created of given max length.")

    x_train, y_train = create_inputs_targets(train_squad_qas)

    with open(conf["dev_path"]) as f:
        dev_json = json.load(f)
    dev_squad_qas = create_squad_qas(dev_json, tokenizer, max_len)
    dev_squad_qas = [x for x in dev_squad_qas if x.skip is False]
    print(f"{len(dev_squad_qas)} evaluation points created of given max length.")

    x_dev, y_dev = create_inputs_targets(dev_squad_qas)

    return x_train, y_train, x_dev, y_dev, train_squad_qas, dev_squad_qas


def process_training(conf, epochs, use_tpu):
    bert_pre_trained = conf["bert_pre_trained"]
    bert_folder = conf["bert_folder"]

    create_bert_tokenizer(bert_pre_trained, bert_folder)
    tokenizer = load_bert_tokenizer(bert_folder)
    x_train, y_train, x_dev, y_dev, train_squad_qas, dev_squad_qas = load_squad_dataset(conf, tokenizer)
    model = create_model(conf, use_tpu)
    model_train(model, conf, epochs, x_train, y_train, x_dev, y_dev, dev_squad_qas)


def main():
    seed = 22
    set_seeds(seed)

    parser = argparse.ArgumentParser(description="Squad question answering Bert Based Tensorflow Model")
    parser.add_argument("--use_tpu", type=bool, default=False, help='To load existing saved model and continue')
    parser.add_argument("--epochs", type=int, default=1, help='To load existing saved model and continue')
    parser.add_argument("--config", type=str, default='bert', help='The basic model configuration dictionary to use.')

    args = parser.parse_args()

    epochs = args.epochs
    use_tpu = args.use_tpu

    if args.config == 'bert':
        conf = config.bert_conf

    process_training(conf, epochs, use_tpu)


if __name__ == "__main__":
    main()




