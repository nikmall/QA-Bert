from pydantic import BaseModel, validator
from typing import List

from config import bert_conf


class ContextQuestion(BaseModel):
    context: str
    question: str


class ContextQuestionList(BaseModel):
    data: List[ContextQuestion]


