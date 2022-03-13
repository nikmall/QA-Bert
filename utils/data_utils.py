import numpy as np


def create_api_inputs(text_question_list):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
    }
    for text_question in text_question_list:
        if text_question.skip is False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(text_question, key))

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x_test = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    return x_test


class ApiQuestion:
    def __init__(self, context, question, max_len):
        self.question = question
        self.context = context
        self.max_len = max_len
        self.skip = False
        self.input_ids = None
        self.token_type_ids = None
        self.attention_mask = None
        self.context_token_to_char = None

    def preprocess(self, tokenizer):
        # Remove white spaces
        context = " ".join(str(self.context).split())
        question = " ".join(str(self.question).split())

        # Tokenize context and question
        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]  # after '[CLS]'
        type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask = [1] * len(input_ids)

        # Pad ids, attention and type  and create attention masks.
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:  # pad
            self.input_ids = input_ids + ([0] * padding_length)
            self.attention_mask = attention_mask + ([0] * padding_length)
            self.token_type_ids = type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.context_token_to_char = tokenized_context.offsets


def create_api_questions(api_data, tokenizer, max_len):
    """
    Reads the API json object of questions and answers, making each context and question-answer(s) into a
    ApiQuestion, preprocessing it calculating input ids, token ids and attention mask for predicting.
    :param api_data: the dictionary of api context-questions data
    :param tokenizer: the Bert Tokenizer object
    :param max_len: max length of model input(context+question)
    :return: a list of preprocessed ApiQuestion objects
    """
    api_questions = []
    for item in api_data["data"]:
        context = item["context"]
        question = item["question"]
        api_question = ApiQuestion(context, question, max_len)
        api_question.preprocess(tokenizer)
        api_questions.append(api_question)
    return api_questions


def create_inputs_targets(squad_question_answers):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for squad_question_answer in squad_question_answers:
        for key in dataset_dict:
            dataset_dict[key].append(getattr(squad_question_answer, key))

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


class SquadQuestionAnswer:
    def __init__(self, question, context, answer_start_idx, answer, all_answers, max_len):
        self.question = question
        self.context = context
        self.answer_start_idx = answer_start_idx
        self.answer = answer
        self.all_answers = all_answers
        self.max_len = max_len
        self.skip = False
        self.start_token_idx = None
        self.end_token_idx = None
        self.input_ids = None
        self.token_type_ids = None
        self.attention_mask = None
        self.context_token_to_char = None

    def preprocess(self, tokenizer):
        # Remove white spaces
        context = " ".join(str(self.context).split())
        question = " ".join(str(self.question).split())
        answer = " ".join(str(self.answer).split())
        answer_start_idx = self.answer_start_idx

        # Calculate answer end character index in context
        answer_end_idx = answer_start_idx + len(answer)
        if answer_end_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(answer_start_idx, answer_end_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        self.start_token_idx = ans_token_idx[0]
        self.end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create input ids
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        # Create type ids
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        # create attention masks
        attention_mask = [1] * len(input_ids)

        # Pad and skip if len exceeds limit
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:  # pad
            self.input_ids = input_ids + ([0] * padding_length)
            self.attention_mask = attention_mask + ([0] * padding_length)
            self.token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.context_token_to_char = tokenized_context.offsets



def create_squad_qas(squad_json, tokenizer, max_len):
    """
    Reads the json squad data file, making each context and question-answer(s) into a
    SquadQuestionAnswer object, preprocess calculating input ids, token ids, attention mask
    start token end token and tokenized offsets.
    :param squad_json: the dictionary of squad data
    :param tokenizer: the Bert Tokenizer object
    :param max_len: max length of model input(context+question)
    :return: a list of preprocessed SquadQuestionAnswer object
    """
    squad_qas = []

    for item in squad_json["data"]:
        for para in item["paragraphs"]:
            for qa in para["qas"]:
                context = para["context"]
                question = qa["question"]
                answer = qa["answers"][0]["text"]
                all_answers = [_["text"] for _ in qa["answers"]]
                answer_start_idx = qa["answers"][0]["answer_start"]

                squad_qa = SquadQuestionAnswer(
                    question, context, answer_start_idx, answer, all_answers, max_len
                )
                squad_qa.preprocess(tokenizer)
                squad_qas.append(squad_qa)
    return squad_qas
