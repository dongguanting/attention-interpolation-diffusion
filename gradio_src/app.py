import os
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

from pipeline_interpolated_sd import InterpolationStableDiffusionPipeline
from pipeline_interpolated_sdxl import InterpolationStableDiffusionXLPipeline
from prior import BetaPriorPipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"

title = r"""
<h1 align="center">PAID: (Prompt-guided) Attention Interpolation of Text-to-Image Diffusion</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/QY-H00/attention-interpolation-diffusion/tree/public' target='_blank'><b>PAID: (Prompt-guided) Attention Interpolation of Text-to-Image Diffusion</b></a>.<br>
How to use:<br>
1. Input prompt 1, prompt 2 and negative prompt.
2. For <b> Compositional Generation </b> Input the guidance prompt and choose the one you are satisfied!
3. For <b> Image morphing </b> Input the image prompt 1 and image prompt 2, and choose IP-Adapter.
4. For <b> Scale Control </b> Input the same text for prompt 1 and prompt 2, leave image prompt 1 blank and upload image prompt 2. Then choose IP-Adapter or IP-Composition-Adapter.
5. <b> Note that the time required for the SD-series with an exploration size of 10 is around 120 seconds. XL-series with an exploration size 5 is around 5 minutes 30 seconds. </b>
6. Click the <b>Generate</b> button to begin generating images.
7. Enjoy! 😊"""

article = r"""
---
✒️ **Citation**
<br>
If you found this demo/our paper useful, please consider citing:
```bibtex
@article{he2024aid,
  title={AID: Attention Interpolation of Text-to-Image Diffusion},
  author={He, Qiyuan and Wang, Jinghao and Liu, Ziwei and Yao, Angela},
  journal={arXiv preprint arXiv:2403.17924},
  year={2024}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue in our <a href='https://github.com/QY-H00/attention-interpolation-diffusion/tree/public' target='_blank'><b>Github Repo</b></a> or directly reach us out at <b>qhe@u.nus.edu.sg</b>.
"""

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = False
USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"
PREVIEW_IMAGES = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = InterpolationStableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V4.0_noVAE",
    torch_dtype=torch.float16
)
pipeline.to(device, dtype=torch.float16)


def change_model_fn(model_name: str) -> None:
    global device
    name_mapping = {
        "AOM3": "hogiahien/aom3",
        "SD1.5-512": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "SD2.1-768": "stabilityai/stable-diffusion-2-1",
        "RealVis-v4.0": "SG161222/Realistic_Vision_V4.0_noVAE",
        "SDXL-1024": "stabilityai/stable-diffusion-xl-base-1.0",
        "Playground-XL-v2": "playgroundai/playground-v2.5-1024px-aesthetic",
        "Juggernaut-XL-v9": "RunDiffusion/Juggernaut-XL-v9"
    }
    if device == torch.device("cpu"):
            dtype = torch.float16
    else:
        dtype = torch.float16
    if "XL" not in model_name:
        globals()["pipeline"] = InterpolationStableDiffusionPipeline.from_pretrained(
            name_mapping[model_name], torch_dtype=dtype
        )
        globals()["pipeline"].to(device, dtype=torch.float16)
    else:
        globals()["pipeline"] = InterpolationStableDiffusionXLPipeline.from_pretrained(
            name_mapping[model_name], torch_dtype=dtype
        )
        globals()["pipeline"].to(device)


def change_adapter_fn(adapter_name: str) -> None:
    global pipeline
    if adapter_name == "IP-Adapter":
        if isinstance(pipeline, InterpolationStableDiffusionPipeline):
            pipeline.load_aid_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        else:
            pipeline.load_aid_ip_adapter("ozzygt/sdxl-ip-adapter", "", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
    elif adapter_name == "IP-Composition-Adapter":
        if isinstance(pipeline, InterpolationStableDiffusionPipeline):
            pipeline.load_aid_ip_adapter("ostris/ip-composition-adapter", subfolder="", weight_name="ip_plus_composition_sd15.safetensors")
        else:
            pipeline.load_aid_ip_adapter("ozzygt/sdxl-ip-adapter", subfolder="", weight_name="ip_plus_composition_sdxl.safetensors")
    else:
        pipeline.load_aid()


def save_image(img, index):
    unique_name = f"{index}.png"
    img = Image.fromarray(img)
    img.save(unique_name)
    return unique_name


def get_example() -> list[list[str | float | int ]]:
    case = [
        [
            "A statue",
            "A dragon",
            "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
            "",
            None,
            None,
            50,
            10,
            5,
            5.0,
            0.5,
            "RealVis-v4.0",
            "None",
            0,
            True,
        ],
        [
            "A photo of a statue",
            "Het meisje met de parel, by Vermeer",
            "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
            "",
            Image.open("asset/statue.jpg"),
            Image.open("asset/vermeer.jpg"),
            50,
            10,
            5,
            5.0,
            0.5,
            "RealVis-v4.0",
            "IP-Adapter",
            0,
            True,
        ],
        [
            "A boy is smiling",
            "A boy is smiling",
            "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
            "",
            None,
            Image.open("asset/vermeer.jpg"),
            50,
            10,
            5,
            5.0,
            0.5,
            "RealVis-v4.0",
            "IP-Composition-Adapter",
            0,
            True,
        ],
        [
            "masterpiece, best quality, very aesthetic, absurdres, A dog",
            "masterpiece, best quality, very aesthetic, absurdres, A car",
            "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
            "masterpiece, best quality, very aesthetic, absurdres, the toy, named 'Dog-Car', is designed as a dog figure with car wheels instead of feet",
            None,
            None,
            50,
            5,
            5,
            5.0,
            0.5,
            "RealVis-v4.0",
            "None",
            1002,
            True
        ],
        [
            "masterpiece, best quality, very aesthetic, absurdres, A dog",
            "masterpiece, best quality, very aesthetic, absurdres, A car",
            "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
            "masterpiece, best quality, very aesthetic, absurdres, a dog is driving a car",
            None,
            None,
            28,
            5,
            5,
            5.0,
            0.5,
            "Playground-XL-v2",
            "None",
            1002,
            True
        ]
        # [
        #     "masterpiece, best quality, very aesthetic, absurdres, A cat is smiling, face portrait",
        #     "masterpiece, best quality, very aesthetic, absurdres, A beautiful lady, face portrait",
        #     "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
        #     None,
        #     None,
        #     None,
        #     28,
        #     7,
        #     5,
        #     5.0,
        #     1.0,
        #     "Playground-XL-v2"
        # ],
        # [
        #     "masterpiece, best quality, very aesthetic, absurdres, A dog",
        #     "masterpiece, best quality, very aesthetic, absurdres, A car",
        #     "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
        #     "masterpiece, best quality, very aesthetic, absurdres, the toy, named 'Dog-Car', is designed as a dog figure with car wheels instead of feet",
        #     None,
        #     None,
        #     28,
        #     5,
        #     5,
        #     5.0,
        #     0.5,
        #     "Playground-XL-v2"
        # ],

    ]
    return case


def change_generate_button_fn(enable: int) -> gr.Button:
    if enable == 0:
        return gr.Button(interactive=False, value="Switching Model...")
    else:
        return gr.Button(interactive=True, value="Generate")


def dynamic_gallery_fn(interpolation_size: int):
    return gr.Gallery(
        label="Result", show_label=False, rows=1, columns=interpolation_size
    )


@torch.no_grad()
def generate(
    prompt1,
    prompt2,
    negative_prompt,
    guide_prompt=None,
    image_prompt1=None,
    image_prompt2=None,
    num_inference_steps=28,
    exploration_size=16,
    interpolation_size=7,
    guidance_scale=5.0,
    warmup_ratio=0.5,
    seed=0,
    same_latent=True,
) -> np.ndarray:
    global pipeline
    global adapter_choice
    beta_pipe = BetaPriorPipeline(pipeline)
    if guide_prompt == "":
        guide_prompt = None
    generator = (
        torch.cuda.manual_seed(seed)
        if torch.cuda.is_available()
        else torch.manual_seed(seed)
    )
    size = pipeline.unet.config.sample_size
    latent1 = torch.randn((1, 4, size, size,), device="cuda", dtype=pipeline.unet.dtype, generator=generator)
    if same_latent:
        latent2 = latent1.clone()
    else:
        latent2 = torch.randn((1, 4, size, size,), device="cuda", dtype=pipeline.unet.dtype, generator=generator)

    if image_prompt1 is None and image_prompt2 is None:
        pipeline.load_aid()
    elif (image_prompt1 is None and image_prompt2 is not None):
        if adapter_choice.value == "IP-Adapter":
            if isinstance(pipeline, InterpolationStableDiffusionPipeline):
                pipeline.load_aid_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
            else:
                pipeline.load_aid_ip_adapter("ozzygt/sdxl-ip-adapter", "", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
        elif adapter_choice.value == "IP-Composition-Adapter":
            if isinstance(pipeline, InterpolationStableDiffusionPipeline):
                pipeline.load_aid_ip_adapter("ostris/ip-composition-adapter", subfolder="", weight_name="ip_plus_composition_sd15.safetensors")
            else:
                pipeline.load_aid_ip_adapter("ozzygt/sdxl-ip-adapter", subfolder="", weight_name="ip_plus_composition_sdxl.safetensors")
    elif (image_prompt1 is None and image_prompt2 is not None):
        if adapter_choice.value == "IP-Adapter":
            if isinstance(pipeline, InterpolationStableDiffusionPipeline):
                pipeline.load_aid_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin", early="scale_control")
            else:
                pipeline.load_aid_ip_adapter("ozzygt/sdxl-ip-adapter", "", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors", early="scale_control")
        elif adapter_choice.value == "IP-Composition-Adapter":
            if isinstance(pipeline, InterpolationStableDiffusionPipeline):
                pipeline.load_aid_ip_adapter("ostris/ip-composition-adapter", subfolder="", weight_name="ip_plus_composition_sd15.safetensors", early="scale_control")
            else:
                pipeline.load_aid_ip_adapter("ozzygt/sdxl-ip-adapter", subfolder="", weight_name="ip_plus_composition_sdxl.safetensors", early="scale_control")
    else:
        raise ValueError("To use scale control, please provide only the right image; To use image morphing, please provide images from both side.")
    images = beta_pipe.generate_interpolation(
        gr.Progress(),
        prompt1,
        prompt2,
        negative_prompt,
        latent1,
        latent2,
        num_inference_steps,
        image_start=image_prompt1,
        image_end=image_prompt2,
        exploration_size=exploration_size,
        interpolation_size=interpolation_size,
        output_type="np",
        guide_prompt=guide_prompt,
        guidance_scale=guidance_scale,
        warmup_ratio=warmup_ratio
    )
    return images


interpolation_size = None

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row(elem_classes="grid-container"):
        with gr.Group():
            with gr.Column(elem_classes="grid-item"):  # 左侧列
                prompt1 = gr.Text(
                    label="Prompt 1",
                    max_lines=3,
                    placeholder="Enter the First Prompt",
                    interactive=True,
                    value="A photo of a cat",
                )
                prompt2 = gr.Text(
                    label="Prompt 2",
                    max_lines=3,
                    placeholder="Enter the Second Prompt",
                    interactive=True,
                    value="A photo of a beautiful lady",
                )
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    max_lines=3,
                    placeholder="Enter a Negative Prompt",
                    interactive=True,
                    value="nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
                )
                guidance_prompt = gr.Text(
                    label="Guidance prompt (Optional)",
                    max_lines=3,
                    placeholder="Enter a Guidance Prompt",
                    interactive=True,
                    value="",
                )

        with gr.Group():
            with gr.Column(elem_classes="grid-item"):  # 右侧列
                with gr.Row(elem_classes="flex-grow"):
                    image_prompt1 = gr.Image(label="Image Prompt 1 (Optional)", interactive=True, height=236, width=235)
                    image_prompt2 = gr.Image(label="Image Prompt 2 (Optional)", interactive=True, height=236, width=235)
                with gr.Row(elem_classes="flex-grow"):
                    model_choice = gr.Dropdown(
                        ["RealVis-v4.0", "SD1.4-512", "SD1.5-512", "SD2.1-768", "AOM3", "SDXL-1024", "Playground-XL-v2", "Juggernaut-XL-v9"],
                        label="Model",
                        value="RealVis-v4.0",
                        interactive=True,
                        info="All series are running on float16; SD2.1 does not support IP-Adapter; XL-Series takes longer time",
                    )
                    adapter_choice = gr.Dropdown(
                        ["None", "IP-Adapter", "IP-Composition-Adapter"],
                        label="IP-Adapter",
                        value="None",
                        interactive=True,
                        info="Only set to IP-Adapter or IP-Composition-Adapter when using image prompt",
                    )

    with gr.Group():
        result = gr.Gallery(label="Result", show_label=False, rows=1, columns=3)
        generate_button = gr.Button(value="Generate", variant="primary")

    with gr.Accordion("Advanced options", open=True):
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    interpolation_size = gr.Slider(
                        label="Interpolation Size",
                        minimum=3,
                        maximum=7,
                        step=1,
                        value=5,
                        info="Interpolation size includes the start and end images",
                    )
                    exploration_size = gr.Slider(
                        label="Exploration Size",
                        minimum=7,
                        maximum=16,
                        step=1,
                        value=10,
                        info="Exploration size has to be larger than interpolation size",
                    )
        with gr.Row():
            with gr.Column():
                warmup_ratio = gr.Slider(
                    label="Warmup Ratio",
                    minimum=0.02,
                    maximum=1,
                    step=0.01,
                    value=0.5,
                    interactive=True,
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=0,
                    maximum=20,
                    step=0.1,
                    value=5.0,
                    interactive=True,
                )
        num_inference_steps = gr.Slider(
            label="Inference Steps",
            minimum=25,
            maximum=50,
            step=1,
            value=50,
            interactive=True,
        )
        with gr.Column():
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            same_latent = gr.Checkbox(
                label="Same latent",
                value=False,
                info="Use the same latent for start and end images",
                show_label=True,
            )

    gr.Examples(
        examples=get_example(),
        inputs=[
            prompt1,
            prompt2,
            negative_prompt,
            guidance_prompt,
            image_prompt1,
            image_prompt2,
            num_inference_steps,
            exploration_size,
            interpolation_size,
            guidance_scale,
            warmup_ratio,
            model_choice,
            adapter_choice,
            seed,
            same_latent,
        ],
        cache_examples=CACHE_EXAMPLES,
    )

    model_choice.change(
        fn=change_generate_button_fn,
        inputs=gr.Number(0, visible=False),
        outputs=generate_button,
    ).then(fn=change_model_fn, inputs=model_choice).then(
        fn=change_generate_button_fn,
        inputs=gr.Number(1, visible=False),
        outputs=generate_button,
    )

    adapter_choice.change(
        fn=change_generate_button_fn,
        inputs=gr.Number(0, visible=False),
        outputs=generate_button,
    ).then(fn=change_adapter_fn, inputs=[adapter_choice]).then(
        fn=change_generate_button_fn,
        inputs=gr.Number(1, visible=False),
        outputs=generate_button,
    )

    inputs = [
        prompt1,
        prompt2,
        negative_prompt,
        guidance_prompt,
        image_prompt1,
        image_prompt2,
        num_inference_steps,
        exploration_size,
        interpolation_size,
        guidance_scale,
        warmup_ratio,
        seed,
        same_latent,
    ]
    generate_button.click(
        fn=dynamic_gallery_fn,
        inputs=interpolation_size,
        outputs=result,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )
    gr.Markdown(article)

demo.launch()
