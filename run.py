import gc
from collections import OrderedDict
from typing import Any

import torch
import rembg
import numpy as np
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

from lib.apis.mvedit import MVEditRunner
from lib.pipelines.utils import (
    init_common_modules,
    rgba_to_rgb,
    do_segmentation,
    do_segmentation_pil,
    pad_rgba_image,
    join_prompts,
    zero123plus_postprocess,
)

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


runner = MVEditRunner(
    device=torch.device("cuda"),
    local_files_only=False,
    unload_models=True,
    out_dir="debug_viz",
    save_interval=1,
    debug=True,
    no_safe=True,
)


def remove_background(
    image: Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def fill_background(image):
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")


if __name__ == "__main__":
    seed = 42
    device = "cuda:0"
    txt2img_pipeline = None

    image_defaults["prompt"] = prompt
    image_defaults["scheduler"] = "DPMSolverMultistep"
    image_defaults["negative_prompt"] = ""
    image_defaults["steps"] = 32
    image_defaults["adapter_ckpt"] = (
        "/home/and/projects/itmo/diploma/sd_fine_tuning/sd-2-1-chairs-lora/checkpoint-4500"
    )
    image_defaults["adapter_filename"] = "model.safetensors"

    # Text to image
    # image = runner.run_text_to_img(SEED, **image_defaults)
    ## Initialize txt2img_pipelineline
    txt2img_pipeline = StableDiffusionPipeline.from_pretrained(
        image_defaults["checkpoint"]
    )
    txt2img_pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        image_defaults["checkpoint"],
        subfolder="scheduler",
        torch_dtype=torch.float16,
    )
    cfg = dict(txt2img_pipeline.scheduler.config)
    if "_use_default_values" in cfg:
        del cfg["_use_default_values"]
    extra_kwargs = dict(use_karras_sigmas=False, timestep_spacing="trailing")
    cfg.update(extra_kwargs)

    ## Load adapter
    txt2img_pipeline.unet.load_attn_procs(
        image_defaults["adapter_ckpt"], weight_name=image_defaults["adapter_filename"]
    )

    txt2img_pipeline.to(device)

    ## Inferencee
    image = txt2img_pipeline(
        height=image_defaults["height"],
        width=image_defaults["width"],
        prompt=join_prompts(image_defaults["prompt"], image_defaults["aux_prompt"]),
        negative_prompt=join_prompts(
            image_defaults["negative_prompt"], image_defaults["aux_negative_prompt"]
        ),
        num_inference_steps=image_defaults["steps"],
        guidance_scale=image_defaults["cfg_scale"],
        return_dict=False,
    )[0][0]

    ## Save result
    image.save("image1.jpg")

    ## Unload pipeline
    txt2img_pipeline = None
    gc.collect()
    torch.cuda.empty_cache()

    # Segment first plan objects
    rembg_session = rembg.new_session()
    image = image.convert("RGB")
    image = remove_background(image, rembg_session)
    image = fill_background(image)

    image.save("image2.jpg")

    # Image to multi-view
    img2mv_pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    )
    img2mv_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        img2mv_pipeline.scheduler.config, timestep_spacing="trailing"
    )
    img2mv_pipeline.to(device)

    mv_images = np.zeros((36, 320, 320, 3), dtype=np.uint8)
    for i in range(6):
        mv_result = img2mv_pipeline(image, num_inference_steps=10).images[0]
        images = (
            np.asarray(mv_result)
            .reshape(3, 320, 2, 320, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(6, 320, 320, 3)
        )
        mv_images[i * 6 : (i + 1) * 6] = images

    # TODO: add background segmentation for each image

    ## Save result
    for i, img in enumerate(mv_images):
        img = Image.fromarray(img).convert("RGB")
        img.save(f"image3_{i}.jpg")

    ## Unload pipeline
    img2mv_pipeline = None
    gc.collect()
    torch.cuda.empty_cache()

    # image = image.convert("RGBA")
    # runner.load_zero123plus_pipeline("sudo-ai/zero123plus-v1.1")
    # image = pad_rgba_image(np.asarray(image), ratio=runner.zero123plus_pad_ratio)
    # images = runner.proc_zero123plus(
    #     seed,
    #     image,
    #     seg_padding=32,
    #     out_margin=160 - runner.zero123plus_crop_half_size,
    # )

    # ## Save result
    # for i, img in enumerate(images):
    #     img = Image.fromarray(img).convert("RGB")
    #     img.save(f"image33_{i}.jpg")

    # ## Unload pipeline
    # img2mv_pipeline = None
    # gc.collect()
    # torch.cuda.empty_cache()

    # Multi-view to 3D
    print_gpu_memory()

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

    # img_to_3d_inputs = (
    #     [var_dict["fg_image"]]
    #     + [var_dict[k] for k in nerf_mesh_defaults.keys() if k not in default_var_dict]
    #     + [
    #         var_dict["superres"][k]
    #         for k in superres_defaults.keys()
    #         if "superres_" + k not in default_superres_var_dict
    #     ]
    #     + var_dict["zero123plus_outputs"]
    # )

    mesh_path = runner.run_zero123plus_to_mesh(
        seed,
        image.convert("RGBA"),
        *nerf_mesh_defaults.values(),
        *{f"superres_{k}": v for k, v in superres_defaults.items()}.values(),
        *[np.asarray(Image.fromarray(img).convert("RGBA")) for img in mv_images],
    )
    print(mesh_path)

    # 3D to video
