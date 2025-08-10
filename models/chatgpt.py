import os
from openai import OpenAI

class ChatGPTModel:
    def __init__(self, system_prompt: str = None, output_tokens: int = 512):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=self.api_key,
        )
        self.output_tokens = output_tokens
        print("ChatGPTModel initialized.")

    def generate(self, prompt: str) -> str:
        system_prompt = self.system_prompt

        completion = self.client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=self.output_tokens,
        )

        output_text = completion.output_text.replace("```json", "").replace("```", "").replace("\\n", "\r\n")

        print("ChatGPTModel response:", output_text)
        return output_text
