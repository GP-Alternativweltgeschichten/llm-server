import os
from dotenv import load_dotenv

load_dotenv()
def get_model():
    model_name = os.getenv("MODEL_NAME")
    if model_name == "mistral":
        hf_login()

        from models.mistral import MistralModel
        return MistralModel()
    elif model_name == "llama_32":
        hf_login()

        from models.llama_32 import LLaMa32Model
        return LLaMa32Model()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def hf_login():
    from huggingface_hub import login
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError("No hugging face token found.")
    login(token=hf_token)
    print("Logged in to Hugging Face Hub successfully.")
