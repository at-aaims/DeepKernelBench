import time
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

#Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# warmup
prompt = "A cat walks on the grass, realistic"
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=6.0
).frames[0]

prompt = "A dog plays with a ball, realistic"
device = torch.device(f"cuda:0")
torch.cuda.reset_max_memory_allocated(device)

start_time = time.perf_counter()
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=6.0
).frames[0]
end_time = time.perf_counter()

peak_bytes = torch.cuda.max_memory_allocated(device)
peak_mem_gb = peak_bytes / (1024**3)

print(f"Elapsed time: {end_time - start_time:.6f} seconds")
print(f"Peak memory usage: {peak_mem_gb:.3f} GB")

# verify
export_to_video(output, "output.mp4", fps=16)
