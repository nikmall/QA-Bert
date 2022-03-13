import re
import string
from collections import Counter


def normalize_answer(text):
    """
    Clean and normalize the text following the standard Squad Dataset process, required for evaluation.
    The normalization includes lowercase, removing punctuations, articles and extra white space.
    :param text: A string
    :return: text: The string cleaned after the normalizing steps.
    """

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def get_raw_scores(all_answers, pred_answer):
    """
    Calculate the f1 and exact match scores of given prediction and true answer(s).
    It follow the standard practise of Squad dataset evaluation for Raw scores. Raw
    means thresholds are not applied.
    :param all_answers: string predicted answer
    :param pred_answer: list of true answer(s)
    :return: The exact_score, f1_score for the given answer
    """
    exact_scores = []
    f1_scores = []
    for answer in all_answers:
        exact_scores.append(compute_exact(answer, pred_answer))
        f1_scores.append(compute_f1(answer, pred_answer))
    exact_score = max(exact_scores)
    f1_score = max(f1_scores)
    return exact_score, f1_score


def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(normalized_pred_answer, normalized_true_answer):
    pred_tokens = normalized_pred_answer.split()
    gold_tokens = normalized_true_answer.split()

    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
