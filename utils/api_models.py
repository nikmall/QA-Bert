from pydantic import BaseModel, validator
from typing import List

from config import bert_conf


class ContextQuestion(BaseModel):
    context: str
    question: str



class ContextQuestionList(BaseModel):
    data: List[ContextQuestion]

    @validator('data')
    def check_combined_length(cls, field_value, values, field, config):
        max_len = bert_conf['max_len']
        for x in field_value:
            if (len(x.context) + len(x.question)) > max_len:
                raise ValueError(f'The combined length of given Context and Question exceeds maximum limit of {max_len}.')
        return field_value