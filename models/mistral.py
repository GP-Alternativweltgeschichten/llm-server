from transformers import AutoModelForCausalLM
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import torch


class MistralModel:
    def __init__(self):
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"

        print("Loading model:", model_id)
        self.tokenizer = MistralTokenizer.v1()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto"
        )

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
```
This is the prompt: 
            """
        )

        completion_request = ChatCompletionRequest(
            messages=[
                UserMessage(content=system_prompt),
                UserMessage(content=prompt)
            ]
        )

        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        input_ids = torch.tensor([tokens])
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids).to(self.model.device)

        generated_ids = self.model.generate(
            input_ids, max_new_tokens=256,
            do_sample=True,
            attention_mask=attention_mask,
        )
        output = self.tokenizer.decode(generated_ids[0].tolist())

        json_start = output.find("[")
        json_end = output.rfind("]") + 1

        if json_start == -1 or json_end == -1:
            return '{"error": "No JSON found in model response"}'

        return output[json_start:json_end]
