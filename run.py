import gc
from collections import OrderedDict
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from lib.apis.runner import Runner

torch.set_grad_enabled(False)


prompt = """Dining chair. Color: Tallmyra black/gray. Cover: Removable, machine washable cotton-polyester blend (100% recycled). Frame: Ash-veneer and solid birch. Leg protectors: Self-adhesive floor protectors (sold separately). Materials: Solid birch legs, layer-glued wood veneer seat rail, plywood seat, and backrest, clear acrylic lacquer. Size: Suitable for all activities around the dining table. Design: K Hagberg/M Hagberg"""

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


def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")


def process(prompt, cache_dir):
    image_defaults["prompt"] = prompt

    runner = Runner(
        device=torch.device("cuda"),
        local_files_only=False,
        unload_models=True,
        out_dir=None,
        save_interval=None,
        debug=False,
    )

    # Text to image
    image = runner.run_text_to_img(seed, *image_defaults.values())
    image.save("image1.png")

    # Segment first plan objects
    image = runner.run_segmentation(image)
    image.save("image2.png")

    # Image to multi-view
    mv_images = runner.run_zero123plus(seed, image)
    # for i, img in enumerate(mv_images):
    #     img = Image.fromarray(img).convert("RGB")
    #     img.save(f"image3_{i}.jpg")

    # Multi-view to 3D
    mesh_path = runner.run_zero123plus_to_mesh(
        seed,
        image.convert("RGBA"),
        *nerf_mesh_defaults.values(),
        *{f"superres_{k}": v for k, v in superres_defaults.items()}.values(),
        *[np.asarray(Image.fromarray(img).convert("RGBA")) for img in mv_images],
        cache_dir=cache_dir,
    )

    runner = None
    gc.collect()
    torch.cuda.empty_cache()

    return mesh_path


def mesh_to_video(mesh_path, cache_dir):
    video_defaults["cache_dir"] = cache_dir

    runner = Runner(
        device=torch.device("cuda"),
        local_files_only=False,
        unload_models=True,
        out_dir=None,
        save_interval=None,
        debug=False,
    )

    video_path = runner.run_mesh_to_video(mesh_path, **video_defaults)

    runner = None
    gc.collect()
    torch.cuda.empty_cache()

    return video_path


# Configs
seed = 42
device = "cuda:0"

# image_defaults["prompt"] = prompt
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


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    data_path = "/home/and/projects/itmo/diploma/data/products/dataset_filtered_with_summary_v0.2.csv"
    df = pd.read_csv(data_path)
    prompts = df.product_summary.values
    mesh_paths = ["" for _ in range(len(df))]
    for i, prompt in enumerate(tqdm(prompts)):
        mesh_paths[i] = process(prompt)
        df["mesh_path"] = mesh_paths
        df.to_csv("data_with_meshes.csv", index=False)

    df = pd.read_csv("data_with_meshes.csv")
    mesh_paths = pd.Series(["outputs/"] * len(df)) + df["mesh_path"]
    video_dir = Path("outputs/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_paths = [""] * len(df)
    for i, mesh_path in enumerate(tqdm(mesh_paths)):
        video_paths[i] = mesh_to_video(mesh_path, video_dir)
        df["video_path"] = video_paths
        df.to_csv("data_with_meshes_and_videos.csv", index=False)

    # mesh_path = process(prompt, ".")
    # print(mesh_to_video(mesh_path, "."))
