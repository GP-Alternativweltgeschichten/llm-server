from transformers import AutoModelForCausalLM
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import torch


class MistralModel:
    def __init__(self, system_prompt: str = None):
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.system_prompt = system_prompt

        print("Loading model:", model_id)
        self.tokenizer = MistralTokenizer.v1()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto"
        )
        print("Ready!")

    def generate(self, prompt: str) -> str:
        completion_request = ChatCompletionRequest(
            messages=[
                UserMessage(content=self.system_prompt),
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
