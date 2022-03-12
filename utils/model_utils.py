from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
import os
import numpy as np
import random
import tensorflow as tf

def create_bert_tokenizer(bert_pre_trained, folder):
    """
    ## Creates a pre-trained BERT tokenizer.
    It uses the  Transformers WordPiece tokenizer.
    """
    # Save the slow pretrained tokenizer
    slow_tokenizer = BertTokenizer.from_pretrained(bert_pre_trained)

    save_path = os.path.abspath(folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        slow_tokenizer.save_pretrained(save_path)

def load_bert_tokenizer(dir):
    """
    Load the saved pre-trained bert WordPiece tokenizer from directory.
    :param dir: The directory inside project directory which contains the saved BertTokenizer vocabulary
    :return: The BertWordPieceTokenizer object
    """
    filepath = os.path.abspath(f"{dir}/vocab.txt")
    tokenizer = BertWordPieceTokenizer(filepath, lowercase=True)
    return tokenizer


def set_seeds(seed):
    """
    Initializes the Seeds in numpy, python random and tensorflow for reproducibility.
    :param seed: int the random seed.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
