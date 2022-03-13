from tensorflow.keras.models import load_model
import numpy as np
from tokenizers import BertWordPieceTokenizer

import config
from utils.data_utils import ApiQuestion, create_api_inputs, create_api_questions
from utils.model_utils import load_bert_tokenizer


def predict_api(test_dict):
    conf = config.bert_conf
    tokenizer = load_bert_tokenizer(conf["bert_folder"])

    text_question_list = create_api_questions(test_dict, tokenizer, conf["max_len"])
    x_test = create_api_inputs(text_question_list)
    model = load_model('model')

    pred_start, pred_end = model.predict(x_test)

    answers = []
    for i, (start, end) in enumerate(zip(pred_start, pred_end)):
        api_question = text_question_list[i]
        offsets = api_question.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        if start >= len(offsets):
            answers.append("")
            continue
        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_char_end = offsets[end][1]
            pred_answer = api_question.context[pred_char_start:pred_char_end]
        else:
            pred_answer = api_question.context[pred_char_start:]
        answers.append(pred_answer)

    print(answers)
