import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

class ShapEModel:
    def __init__(self, output_path):
        self.output_path = output_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_model = load_model('text300M', device=self.device)
        self.mesh_model = load_model('transmitter', device=self.device)
        print(type(self.text_model))
        print(type(self.mesh_model))
        self.diffusion = diffusion_from_config(load_config('diffusion'))
        print("ShapE initialized.")

    def generate(self, prompt: str) -> str:
        latents = sample_latents(
            batch_size=1,
            model=self.text_model,
            diffusion=self.diffusion,
            guidance_scale=15.0,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        for latent in latents:
            mesh = decode_latent_mesh(self.mesh_model, latent).tri_mesh()
            with open(self.output_path, 'w') as f:
                mesh.write_obj(f)

        return self.output_path
