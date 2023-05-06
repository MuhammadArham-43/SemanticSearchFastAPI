from pydantic import BaseModel

class DocumentSimilarityModel(BaseModel):
    text_1 : str
    text_2 : str