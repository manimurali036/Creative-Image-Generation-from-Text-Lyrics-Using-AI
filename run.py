!pip install -q diffusers transformers accelerate safetensors torch torchvision 
from diffusers import StableDiffusionPipeline 
import torch 
from PIL import Image 
import IPython.display as display 
model_id = "runwayml/stable-diffusion-v1-5" 
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16 if 
torch.cuda.is_available() else torch.float32 ) 
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu") 
print(",^, Example inputs:") 
print(" - 'I see trees of green, red roses too, I see them bloom for me and you'") 
print(" - 'Cause baby you're a firework, come on let your colors burst'") 
print(" - 'We're soaring, flying, there's not a star in heaven that we can't reach'") 
lyric = input("\nEnter a line of song lyrics: ") 
art_style = "digital art, cinematic lighting, detailed, emotional, surreal" 
prompt = f"An artistic visualization of the lyric: '{lyric}', {art_style}" 
print(f"\n˙·.·Q Generating Art for: {lyric}\n") 
image = pipe(prompt, guidance_scale=8.5, num_inference_steps=30).images[0] 
display.display(image) 
save_path = "lyric2art_output.png" 
image.save(save_path) 
print(f"⬛  Image saved as: {save_path}") 
