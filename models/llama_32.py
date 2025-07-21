from transformers import pipeline
import torch

class LLaMa32Model:
    def __init__(self):
        model_id = "meta-llama/Llama-3.2-1B-Instruct"

        print("Loading model:", model_id)
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
        print("Ready!")


    def generate(self, prompt: str) -> str:
        system_prompt = (
            """You are an expert in urban structures and planning that only responds in JSON format. 
            Center to the task will be modifying the city-scape of the town of Olpe in Germany. 
            I will provide you with information about this town.
            Your task is to transform the given prompt into 'actions' that modify the city-scape of Olpe in the 
            specified bounds.
            Here's an example:
            ```json
            [
  {
    "action": "add",
    "object_type": "park",
    "size": "small",
    "features": [
      "playground"
    ],
    "location": {
      "latitude": 51.02654330098404,
      "longitude": 7.847252148369939
    }
  },
  {
    "action": "remove",
    "object_type": "building",
    "location": {
      "latitude": 51.02654330098404,
      "longitude": 7.847252148369939
    }
  }
]

This is the prompt: 
            """
        )

        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        outputs = self.pipe(messages, max_new_tokens=512)

        output = outputs[0]["generated_text"]
        print("Text:", output[-1])
        output_text = output[-1]["content"].replace("```json", "").replace("```", "").replace("\\n", "\r\n")

        return output_text
