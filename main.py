from fastapi import FastAPI
import uvicorn

from models.docModel import DocumentSimilarityModel

from inference.qqd import qqd_model
from inference.utils import check_similarity, tokenizer

app = FastAPI()

@app.post('/docSim/bert/qqd')
def docSimBertQQD(query: DocumentSimilarityModel):
    similarity = check_similarity(
        trained_model=qqd_model,
        tokenizer=tokenizer,
        question1=query.text_1,
        question2=query.text_2
    )
    
    return {'similarity': similarity}


if __name__ == "__main__":
    uvicorn.run('main:app', port=5000, log_level='info', reload=True)