from transformers import pipeline
import torch

class LLaMa32Model:
    def __init__(self, system_prompt: str = None):
        model_id = "meta-llama/Llama-3.2-1B-Instruct"

        self.system_prompt = system_prompt
        print("Loading model:", model_id)
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
        print("Ready!")


    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        outputs = self.pipe(messages, max_new_tokens=512)

        output = outputs[0]["generated_text"]
        output_text = output[-1]["content"].replace("```json", "").replace("```", "").replace("\\n", "\r\n")

        return output_text
