import os
from dotenv import load_dotenv

load_dotenv()


def get_model():
    model_name = os.getenv("MODEL_NAME")
    system_prompt = """You are an expert in urban structures and planning that only responds in JSON format. 
            Center to the task will be modifying the city-scape of the town of Olpe in Germany.
            Your task is to transform the given prompt into 'actions' that modify the city-scape of Olpe. 
            The user has selected a couple of buildings and spots, then entered a prompt that describes the desired changes to the city.
            You can either add or remove objects. When adding objects, the "prompt" field will be forwarded to a 3D model generation model. 
            Please provide a short and precise description of the object to be added.
            Here's an example:
            ```json
            [
  {
    "action": "add",
    "prompt": "A public park with a playground and benches",
  },
  {
    "action": "remove",
    "id": "building_123",
  }
]
```
This is the prompt: 
            """
    if model_name == "mistral":
        hf_login()

        from models.mistral import MistralModel
        return MistralModel(system_prompt)
    elif model_name == "llama_32":
        hf_login()

        from models.llama_32 import LLaMa32Model
        return LLaMa32Model(system_prompt)
    elif model_name == "chatgpt":
        from models.chatgpt import ChatGPTModel
        return ChatGPTModel(system_prompt)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_blender_model():
    system_prompt = """You are an expert in 3D modeling using Blender's Python API. 
You only respond with clean, minimal Python code that can be executed in Blender 4.0+ using --background --python.
Do not include comments or explanations, do not print markdown (i.e. a leading `python` marker) or any other formatting.
Generate simple low-poly models suitable for a top-down city map view (e.g. buildings, trees, streets).
Use primitive shapes and basic transformations (scale, location).
Always create at least 2 separate mesh objects using primitive shapes (cube, cylinder, cone, uv_sphere) with distinct sizes and locations.
Apply simple materials with different diffuse colors to each object.
Do not join objects.
Ensure all objects are positioned relative to the origin and scaled for Unity import (units in meters).
Always import the "sys" module and always end your script with an export to filepath=sys.argv[-1]:
bpy.ops.export_scene.fbx(filepath=sys.argv[-1])
"""
    model_name = os.getenv("MODEL_NAME")
    output_tokens = 1024

    if model_name == "mistral":
        hf_login()
        from models.mistral import MistralModel
        return MistralModel(system_prompt, output_tokens=output_tokens)
    elif model_name == "llama_32":
        hf_login()
        from models.llama_32 import LLaMa32Model
        return LLaMa32Model(system_prompt, output_tokens=output_tokens)
    elif model_name == "chatgpt":
        from models.chatgpt import ChatGPTModel
        return ChatGPTModel(system_prompt, output_tokens=output_tokens)
    elif model_name == "blender_code_gen":
        from models.chatgpt import ChatGPTModel
        return ChatGPTModel(system_prompt, output_tokens=output_tokens)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_3d_model():
    # model_name = os.getenv("3D_MODEL_NAME")
    from models.shap_e import ShapEModel
    return ShapEModel(output_path="./output/generated_model.obj")

def hf_login():
    from huggingface_hub import login
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError("No hugging face token found.")
    login(token=hf_token)
    print("Logged in to Hugging Face Hub successfully.")
