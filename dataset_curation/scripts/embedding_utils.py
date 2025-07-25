from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
import numpy as np
from dotenv import load_dotenv
import os
from typing import List, Union
load_dotenv()

# MODELS TO CHOOSE:
ESM2 = "facebook/esm2_t6_8M_UR50D" # really light
PROTT5 = "Rostlab/prot_t5_xl_uniref50" # really heavy
SMALL_OPENAI_MODEL = "text-embedding-3-small"
LARGE_OPENAI_MODEL = "text-embedding-3-large"

# openai api key
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_AAseq_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    return model, tokenizer

class FreeTXTEmbedder:
    
    def __init__(self) -> None:
        self.client = OpenAI(api_key=_OPENAI_API_KEY)

    def request_embedding_for(self, inp: Union[str, List[str]], model: str) -> List[np.ndarray]:
        response = self.client.embeddings.create(input=inp, model=model)
        # extract list of embedding class instances (each has 'embedding' field -- what we need)
        data_points = response.data
        
        # convert to numpy arrays
        return [np.array(i.embedding) for i in data_points]


if __name__ == "__main__":
    pass
