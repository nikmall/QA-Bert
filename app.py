import uvicorn
from fastapi import FastAPI
from predicting import predict_api

from utils.api_models import ContextQuestionList


""" 
Run locally using http://localhost:5000
"""

app = FastAPI(
    title="Question Answering Service",
    description="This API predicts answers from given context.",
    version="1.0",
)


@app.get('/health/')
def health():
    return {'status': 'OK'}


@app.post("/predict_questions")
def predict_questions(data: ContextQuestionList):
    answers = predict_api(data)
    return {
        'answers': answers
    }


def main():
    uvicorn.run(app,
                host='0.0.0.0',
                port=5000,
                debug=False,
                )


if __name__ == '__main__':
    main()
