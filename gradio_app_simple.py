import os
import random
import sys
from functools import partial
import shutil
import os.path as osp
import argparse
from collections import OrderedDict

sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "16"

import torch
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from lib.core.mvedit_webui.gradio_custommodel3d import CustomModel3D

from lib.core.mvedit_webui.shared_opts import send_to_click
from lib.core.mvedit_webui.tab_img_to_3d import create_interface_img_to_3d
from lib.core.mvedit_webui.tab_text_to_img_to_3d import (
    create_interface_text_to_img_to_3d,
)
from lib.core.mvedit_webui.tab_3d_to_video import create_interface_3d_to_video
from lib.apis.mvedit import MVEditRunner

image_defaults = OrderedDict(
    [
        ("width", 512),
        ("height", 512),
        ("prompt", None),
        ("negative_prompt", None),
        ("scheduler", None),
        ("steps", None),
        ("cfg_scale", 7),
        ("checkpoint", "stabilityai/stable-diffusion-2-1-base"),
        ("aux_prompt", "best quality, sharp focus, photorealistic, extremely detailed"),
        (
            "aux_negative_prompt",
            "text, watermark, worst quality, low quality, depth of field, blurry, out of focus, low-res, "
            "illustration, painting, drawing",
        ),
        ("adapter_ckpt", None),
        ("adapter_filename", None),
    ]
)


nerf_mesh_defaults = OrderedDict(
    [
        ("prompt", None),
        ("negative_prompt", None),
        ("scheduler", None),
        ("steps", None),
        ("denoising_strength", None),
        ("random_init", None),
        ("cfg_scale", 7),
        ("checkpoint", "runwayml/stable-diffusion-v1-5"),
        # ("checkpoint", "stabilityai/stable-diffusion-2-1-base"),
        ("max_num_views", 32),
        ("aux_prompt", "best quality, sharp focus, photorealistic, extremely detailed"),
        (
            "aux_negative_prompt",
            "text, watermark, worst quality, low quality, depth of field, blurry, out of focus, low-res, "
            "illustration, painting, drawing",
        ),
        ("diff_bs", None),
        ("patch_size", 128),
        ("patch_bs_nerf", 1),
        ("render_bs", 6),
        ("patch_bs", 8),
        ("alpha_soften", 0.02),
        ("normal_reg_weight", 4.0),
        ("start_entropy_weight", 0.0),
        ("end_entropy_weight", 4.0),
        ("entropy_d", 0.015),
        ("mesh_smoothness", 1.0),
        ("n_inverse_steps", None),
        ("init_inverse_steps", None),
        ("tet_init_inverse_steps", 120),
        ("start_lr", 0.01),
        ("end_lr", 0.005),
        ("tet_resolution", None),
    ]
)

superres_defaults = OrderedDict(
    [
        ("do_superres", None),
        ("scheduler", None),
        ("steps", None),
        ("denoising_strength", None),
        ("random_init", None),
        ("cfg_scale", 7),
        ("checkpoint", "runwayml/stable-diffusion-v1-5"),
        # ("checkpoint", "stabilityai/stable-diffusion-2-1-base"),
        ("aux_prompt", "best quality, sharp focus, photorealistic, extremely detailed"),
        (
            "aux_negative_prompt",
            "text, watermark, worst quality, low quality, depth of field, blurry, out of focus, low-res, "
            "illustration, painting, drawing",
        ),
        ("patch_size", 512),
        ("patch_bs", 1),
        ("n_inverse_steps", None),
        ("start_lr", 0.01),
        ("end_lr", 0.01),
    ]
)

video_defaults = OrderedDict(
    [
        ("front_view_id", None),
        ("layer", "RGB"),
        ("distance", 4),
        ("length", 5),
        ("elevation", 10),
        ("fov", 30),
        ("resolution", 512),
        ("fps", 30),
        ("render_bs", 8),
        ("cache_dir", None),
    ]
)

image_defaults["scheduler"] = "DPMSolverMultistep"
image_defaults["negative_prompt"] = "cheap, low-cost, aged, discounted"
image_defaults["steps"] = 50
image_defaults["adapter_ckpt"] = (
    "/home/and/projects/itmo/diploma/sd_fine_tuning/sd-2-1-chairs-lora/checkpoint-4500"
)
image_defaults["adapter_filename"] = "model.safetensors"

nerf_mesh_defaults["prompt"] = ""
nerf_mesh_defaults["negative_prompt"] = ""
nerf_mesh_defaults["scheduler"] = "DPMSolverMultistep"
nerf_mesh_defaults["steps"] = 24
nerf_mesh_defaults["denoising_strength"] = 0.5
nerf_mesh_defaults["random_init"] = False
nerf_mesh_defaults["diff_bs"] = 2
nerf_mesh_defaults["n_inverse_steps"] = 80
nerf_mesh_defaults["init_inverse_steps"] = 640
nerf_mesh_defaults["tet_resolution"] = 128

superres_defaults["do_superres"] = False
superres_defaults["scheduler"] = "DPMSolverSDEKarras"
superres_defaults["steps"] = 24
superres_defaults["denoising_strength"] = 0.4
superres_defaults["random_init"] = False
superres_defaults["n_inverse_steps"] = 80


def parse_args():
    parser = argparse.ArgumentParser(description="MVEdit 3D Toolbox")
    parser.add_argument("--diff-bs", type=int, default=4, help="Diffusion batch size")
    parser.add_argument(
        "--advanced", action="store_true", help="Show advanced settings"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Save debug images to ./viz"
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load local model weights and configuration files",
    )
    parser.add_argument(
        "--no-safe", action="store_true", help="Disable safety checker to free VRAM"
    )
    parser.add_argument(
        "--empty-cache", action="store_true", help="Empty the cache directory"
    )
    parser.add_argument(
        "--unload-models",
        action="store_true",
        help="Auto-unload unused models to free VRAM",
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = "./"

    if args.empty_cache:
        if osp.exists("./gradio_cached_examples"):
            shutil.rmtree("./gradio_cached_examples")
        if os.environ.get("GRADIO_TEMP_DIR", None) is not None and osp.exists(
            os.environ["GRADIO_TEMP_DIR"]
        ):
            shutil.rmtree(os.environ["GRADIO_TEMP_DIR"])

    torch.set_grad_enabled(False)
    runner = MVEditRunner(
        device=torch.device("cuda"),
        local_files_only=args.local_files_only,
        unload_models=args.unload_models,
        out_dir=osp.join(osp.dirname(__file__), "viz") if args.debug else None,
        save_interval=1 if args.debug else None,
        debug=args.debug,
        no_safe=args.no_safe,
    )

    def text_to_image(seed, prompt, negative_prompt):
        image_defaults["prompt"] = prompt
        image_defaults["negative_prompt"] = negative_prompt
        image = runner.run_text_to_img(seed, *image_defaults.values())
        image = runner.run_segmentation(image)
        return image

    def image_to_3d(seed, image, cache_dir="."):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        pbar = gr.Progress().tqdm(None, total=6 + 13)
        mv_images = runner.run_zero123plus(seed, image, pbar=pbar)
        mesh_path = runner.run_zero123plus_to_mesh(
            seed,
            image.convert("RGBA"),
            *nerf_mesh_defaults.values(),
            *{f"superres_{k}": v for k, v in superres_defaults.items()}.values(),
            *[np.asarray(Image.fromarray(img).convert("RGBA")) for img in mv_images],
            cache_dir=cache_dir,
            pbar=pbar,
        )
        return mesh_path

    with gr.Blocks(
        analytics_enabled=False,
        title="3D chairs generation",
        theme=gr.themes.Base(),
    ) as interface:
        md_txt = "# 3D Chairs generation demo app"
        gr.Markdown(md_txt)
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Prompt")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", placeholder="Negative Prompt"
                )
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    min_width=100,
                    precision=0,
                    minimum=-1,
                    maximum=2**31,
                    interactive=True,
                    elem_classes=["force-hide-container"],
                )
                btn_text_to_image = gr.Button("RUN text to image")
                btn_image_to_3d = gr.Button("RUN image to 3D")
                # btn_3d_to_video = gr.Button("RUN 3D to Video")
                btn_generate_all = gr.Button("Generate ALL", variant="primary")

            with gr.Column():
                image_output = gr.Image(label="Image", image_mode="RGBA", height=350)
                model_output = CustomModel3D(
                    height=350,
                    label="3D model",
                    camera_position=(180, 80, 3.0),
                    interactive=False,
                )
                # video_output = gr.Video(label="Video")

        btn_text_to_image.click(
            fn=text_to_image,
            inputs=[
                seed,
                prompt,
                negative_prompt,
            ],
            outputs=image_output,
        )
        btn_image_to_3d.click(
            fn=image_to_3d,
            inputs=[seed, image_output],
            outputs=model_output,
        )

        # btn_3d_to_video.click(
        #     fn=three_d_to_video, inputs=model_output, outputs=video_output
        # )
        btn_generate_all.click(
            fn=text_to_image,
            inputs=[
                seed,
                prompt,
                negative_prompt,
            ],
            outputs=image_output,
        ).success(
            fn=image_to_3d,
            inputs=[seed, image_output],
            outputs=model_output,
        )

        interface.queue().launch(share=args.share, debug=args.debug)


if __name__ == "__main__":
    main()
