import numpy as np
import config

from tensorflow.keras.models import load_model

from utils.data_utils import create_api_inputs, create_api_questions, create_squad_qas, create_inputs_targets
from utils.model_utils import load_bert_tokenizer


def predict_api(test_dict):
    conf = config.bert_conf
    tokenizer = load_bert_tokenizer(conf["bert_folder"])
    # Create data
    data_dict = test_dict.dict()
    text_question_list = create_api_questions(data_dict, tokenizer, conf["max_len"])
    x_test = create_api_inputs(text_question_list)
    model = load_model('model')

    # Return [] in case of empty list after processing(due to max length filtering)
    if x_test[-1].shape[0] == 0:
        return []

    pred_start, pred_end = model.predict(x_test)
    answers = get_pred_answer_text(pred_start, pred_end, text_question_list)
    return answers


def get_pred_answer_text(pred_start, pred_end, questions_list):
    answers = []
    for i, (start, end) in enumerate(zip(pred_start, pred_end)):
        api_question = questions_list[i]
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
    return answers
