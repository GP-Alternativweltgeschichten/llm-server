import os
from dotenv import load_dotenv

load_dotenv()
def get_model():
    model_name = os.getenv("MODEL_NAME")
    if model_name == "mistral":
        from huggingface_hub import login
        from models.mistral import MistralModel

        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_token:
            raise ValueError("No hugging face token found.")

        login(token=hf_token)

        return MistralModel()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
