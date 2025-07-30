import os
from openai import OpenAI

class ChatGPTModel:
    def __init__(self, system_prompt: str = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=self.api_key,
        )
        print("ChatGPTModel initialisiert.")

    def generate(self, prompt: str) -> str:
        system_prompt = self.system_prompt

        completion = self.client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=512
        )

        output_text = completion.output_text.replace("```json", "").replace("```", "").replace("\\n", "\r\n")

        return output_text
