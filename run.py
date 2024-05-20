import gc
import os
import uuid
from collections import OrderedDict
from typing import Any
from functools import partial
from pathlib import Path
import json

import torch
import rembg
import numpy as np
from PIL import Image
import diffusers
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

from lib.apis.mvedit import MVEditRunner
from lib.pipelines import (
    MVEdit3DPipeline,
    MVEditTextureSuperResPipeline,
    MVEditTexturePipeline,
    Zero123PlusPipeline,
)
from lib.models.architecture.ip_adapter import IPAdapter
from lib.models.autoencoders.base_mesh import Mesh, preprocess_mesh
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


def run_zero123plus_to_mesh(device, seed, in_img, *args, cache_dir=None, **kwargs):
    from tqdm.auto import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        StableDiffusionControlNetPipeline,
        DPMSolverMultistepScheduler,
        DPMSolverSDEScheduler,
    )
    from diffusers.models import ControlNetModel
    from diffusers.pipelines.controlnet import MultiControlNetModel
    from diffusers.schedulers import KarrasDiffusionSchedulers
    from omnidata_modules.midas.dpt_depth import DPTDepthModel
    from mmcv.runner.checkpoint import _load_checkpoint
    from mmcv.runner import set_random_seed
    from lib.core.utils.camera_utils import (
        get_pose_from_angles,
        random_surround_views,
        get_pose_from_angles_np,
        view_prompts,
    )
    from lib.core.utils.pose_estimation import (
        init_matcher,
        elev_estimation,
        pose5dof_estimation,
    )
    from lib.pipelines.mvedit_3d_pipeline import default_max_num_views
    from lib.core.mvedit_webui.parameters import parse_3d_args

    def load_normal_model():
        print("\nLoading normal model...")
        normal_model = DPTDepthModel(
            backbone="vitb_rn50_384", num_channels=3
        )  # DPT Hybrid
        checkpoint = _load_checkpoint(
            "huggingface://clay3d/omnidata/omnidata_dpt_normal_v2.ckpt",
            map_location="cpu",
        )
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        normal_model.load_state_dict(state_dict)
        normal_model.to(device=device, dtype=torch.float16)
        print("Normal model loaded.")
        return normal_model

    def load_matcher():
        print("\nLoading feature matcher...")
        matcher = init_matcher().to(device)
        print("Feature matcher loaded.")
        return matcher

    def load_stable_diffusion(stable_diffusion_checkpoint):
        print("\nLoading Stable Diffusion...")
        vae = AutoencoderKL.from_pretrained(
            stable_diffusion_checkpoint,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            stable_diffusion_checkpoint,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            stable_diffusion_checkpoint,
            subfolder="tokenizer",
            torch_dtype=torch.float16,
        )
        unet = UNet2DConditionModel.from_pretrained(
            stable_diffusion_checkpoint,
            subfolder="unet",
            torch_dtype=torch.float16,
        )

        # LOAD LORA
        if "stabilityai/stable-diffusion-2-1-base" in stable_diffusion_checkpoint:
            ckpt_dir = "/home/and/projects/itmo/diploma/sd_fine_tuning/sd-2-1-chairs-lora/checkpoint-4500"
            lora_filename = "model.safetensors"
            unet.load_attn_procs(ckpt_dir, weight_name=lora_filename)

        vae.to(device)
        text_encoder.to(device)
        unet.to(device)
        print("Stable Diffusion loaded.")

        return vae, text_encoder, tokenizer, unet

    def load_scheduler(stable_diffusion_checkpoint, scheduler_type):
        print("\nLoading scheduler...")
        if scheduler_type.endswith("Karras"):
            extra_kwargs = dict(use_karras_sigmas=True, timestep_spacing="leading")
            scheduler_class = scheduler_type[:-6]
        else:
            extra_kwargs = dict(use_karras_sigmas=False, timestep_spacing="trailing")
            scheduler_class = scheduler_type
        sampler_class = getattr(diffusers.schedulers, scheduler_class + "Scheduler")
        scheduler = sampler_class.from_pretrained(
            stable_diffusion_checkpoint,
            subfolder="scheduler",
            torch_dtype=torch.float16,
        )
        cfg = dict(scheduler.config)
        if "_use_default_values" in cfg:
            del cfg["_use_default_values"]
        cfg.update(extra_kwargs)
        scheduler = sampler_class.from_config(cfg)
        print("Scheduler loaded.")
        return scheduler

    def load_ip_adapter(pipe):
        for module in [controlnet, controlnet_depth]:
            if module is not None:
                module.set_use_memory_efficient_attention_xformers(
                    not hasattr(torch.nn.functional, "scaled_dot_product_attention")
                )
        pipe.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        print("\nLoading IP-Adapter...")
        ip_adapter = IPAdapter(
            pipe,
            "huggingface://h94/IP-Adapter/models/ip-adapter-plus_sd15.bin",
            "h94/IP-Adapter",
            device=device,
            dtype=torch.float16,
        )
        print("IP-Adapter loaded.")
        gc.collect()
        return ip_adapter

    def proc_nerf_mesh(
        pipe,
        seed,
        nerf_mesh_kwargs,
        superres_kwargs,
        front_azi=None,
        camera_poses=None,
        use_reference=False,
        use_normal=False,
        **kwargs,
    ):
        print(nerf_mesh_kwargs)
        set_random_seed(seed, deterministic=True)
        prompts = (
            nerf_mesh_kwargs["prompt"]
            if front_azi is None
            else [
                join_prompts(nerf_mesh_kwargs["prompt"], view_prompt)
                for view_prompt in view_prompts(camera_poses, front_azi)
            ]
        )
        out_mesh, ingp_states = pipe(
            prompt=prompts,
            negative_prompt=nerf_mesh_kwargs["negative_prompt"],
            camera_poses=camera_poses,
            use_reference=use_reference,
            use_normal=use_normal,
            guidance_scale=nerf_mesh_kwargs["cfg_scale"],
            num_inference_steps=nerf_mesh_kwargs["steps"],
            denoising_strength=(
                None
                if nerf_mesh_kwargs["random_init"]
                else nerf_mesh_kwargs["denoising_strength"]
            ),
            patch_size=nerf_mesh_kwargs["patch_size"],
            patch_bs=nerf_mesh_kwargs["patch_bs"],
            diff_bs=nerf_mesh_kwargs["diff_bs"],
            render_bs=nerf_mesh_kwargs["render_bs"],
            n_inverse_rays=nerf_mesh_kwargs["patch_size"] ** 2
            * nerf_mesh_kwargs["patch_bs_nerf"],
            n_inverse_steps=nerf_mesh_kwargs["n_inverse_steps"],
            init_inverse_steps=nerf_mesh_kwargs["init_inverse_steps"],
            tet_init_inverse_steps=nerf_mesh_kwargs["tet_init_inverse_steps"],
            default_prompt=nerf_mesh_kwargs["aux_prompt"],
            default_neg_prompt=nerf_mesh_kwargs["aux_negative_prompt"],
            alpha_soften=nerf_mesh_kwargs["alpha_soften"],
            normal_reg_weight=lambda p: nerf_mesh_kwargs["normal_reg_weight"] * (1 - p),
            entropy_weight=lambda p: nerf_mesh_kwargs["start_entropy_weight"]
            + (
                nerf_mesh_kwargs["end_entropy_weight"]
                - nerf_mesh_kwargs["start_entropy_weight"]
            )
            * p,
            bg_width=nerf_mesh_kwargs["entropy_d"],
            mesh_normal_reg_weight=nerf_mesh_kwargs["mesh_smoothness"],
            lr_schedule=lambda p: nerf_mesh_kwargs["start_lr"]
            + (nerf_mesh_kwargs["end_lr"] - nerf_mesh_kwargs["start_lr"]) * p,
            tet_resolution=nerf_mesh_kwargs["tet_resolution"],
            bake_texture=not superres_kwargs["do_superres"],
            prog_bar=tqdm,
            out_dir=None,
            save_interval=None,
            save_all_interval=None,
            mesh_reduction=128 / nerf_mesh_kwargs["tet_resolution"],
            max_num_views=partial(
                default_max_num_views,
                start_num=nerf_mesh_kwargs["max_num_views"],
                mid_num=nerf_mesh_kwargs["max_num_views"] // 2,
            ),
            debug=False,
            **kwargs,
        )
        return out_mesh, ingp_states

    zero123plus_pad_ratio = 0.75
    zero123plus1_2_pad_ratio = 0.9
    zero123plus_crop_ratio = 0.9
    zero123plus_crop_half_size = int(round(160 * zero123plus_crop_ratio))
    zero123plus_acutual_crop_ratio = zero123plus_crop_half_size / 160
    zero123plus_superres_camera_distance = 3.1
    zero123plus_superres_min_elev = 0.0
    zero123plus_superres_max_elev = 0.4
    zero123plus_superres_fov = 40
    zero123plus_superres_num_cameras = 6

    nerf_mesh_kwargs, superres_kwargs, init_images = parse_3d_args(list(args), kwargs)

    normal_model = load_normal_model()
    matcher = load_matcher()
    vae, text_encoder, tokenizer, unet = load_stable_diffusion(
        nerf_mesh_kwargs["checkpoint"]
    )
    scheduler = load_scheduler(
        nerf_mesh_kwargs["checkpoint"], nerf_mesh_kwargs["scheduler"]
    )

    (
        image_enhancer,
        mesh_renderer,
        segmentation,
        nerf,
        tonemapping,
        controlnet,
        controlnet_depth,
    ) = init_common_modules(device)

    pipe = MVEdit3DPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=[controlnet, controlnet_depth],
        scheduler=scheduler,
        nerf=nerf,
        mesh_renderer=mesh_renderer,
        image_enhancer=image_enhancer,
        segmentation=segmentation,
        normal_model=normal_model,
        tonemapping=tonemapping,
    )
    ip_adapter = load_ip_adapter(pipe)

    print(f"\nRunning Zero123++ to mesh with seed {seed}...")

    in_img = pad_rgba_image(
        np.asarray(np.asarray(in_img)),
        ratio=zero123plus_pad_ratio / zero123plus_acutual_crop_ratio,
    )
    focal = 350
    fov = np.rad2deg(np.arctan(zero123plus_crop_half_size / focal) * 2)
    camera_distance = 1 / np.sin(np.radians(fov / 2))
    azims = [30, 90, 150, 210, 270, 330, 330, 270, 210, 150, 90, 30] * 3
    elevs = [30, -20] * 18
    camera_poses = get_pose_from_angles(
        torch.tensor(azims, dtype=torch.float32) * np.pi / 180,
        torch.tensor(elevs, dtype=torch.float32) * np.pi / 180,
        camera_distance,
    )[:, :3].to(device)
    intrinsics = torch.tensor(
        [
            focal,
            focal,
            zero123plus_crop_half_size,
            zero123plus_crop_half_size,
        ],
        dtype=torch.float32,
        device=device,
    )
    intrinsics_size = zero123plus_crop_half_size * 2

    elev, in_pose = elev_estimation(
        Image.fromarray(rgba_to_rgb(in_img, bg_color=(127, 127, 127))),
        init_images,  # [init_images[i] for i in [0, 5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 35]],
        camera_poses,  # camera_poses[[0, 5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 35]],
        intrinsics,
        intrinsics_size,
        matcher,
    )
    init_images = [in_img] + list(init_images)
    camera_poses = torch.cat([in_pose[None, :3], camera_poses], dim=0)

    out_mesh, ingp_states = proc_nerf_mesh(
        pipe,
        seed,
        nerf_mesh_kwargs,
        superres_kwargs,
        init_images=init_images,
        camera_poses=camera_poses,
        intrinsics=intrinsics,
        intrinsics_size=intrinsics_size,
        cam_weights=[3.0] + [1.5, 0.95, 0.93, 0.88, 1.0, 1.45] * 6,
        seg_padding=80,
        keep_views=[0],
        ip_adapter=ip_adapter,
        use_reference=True,
        use_normal=True,
    )

    # if superres_kwargs["do_superres"]:
    #     self.load_stable_diffusion(superres_kwargs["checkpoint"])
    #     self.load_scheduler(superres_kwargs["checkpoint"], superres_kwargs["scheduler"])
    #     pipe = MVEditTextureSuperResPipeline(
    #         vae=self.vae,
    #         text_encoder=self.text_encoder,
    #         tokenizer=self.tokenizer,
    #         unet=self.unet,
    #         controlnet=[self.controlnet, self.controlnet_depth],
    #         scheduler=self.scheduler,
    #         nerf=self.nerf,
    #         mesh_renderer=self.mesh_renderer,
    #     )
    #     ref_pose = get_pose_from_angles(
    #         torch.zeros((1,), dtype=torch.float32, device=device),
    #         in_pose.new_tensor([(elev + 10) * np.pi / 180]),
    #         in_pose.new_tensor([camera_distance]),
    #     )[0, :3]
    #     if self.empty_cache:
    #         torch.cuda.empty_cache()
    #     out_mesh = self.proc_texture_superres(
    #         pipe,
    #         seed + 1 if seed < 2**31 else 0,
    #         out_mesh,
    #         ingp_states,
    #         nerf_mesh_kwargs,
    #         superres_kwargs,
    #         camera_distance=self.zero123plus_superres_camera_distance,
    #         fov=self.zero123plus_superres_fov,
    #         num_cameras=6,
    #         min_elev=self.zero123plus_superres_min_elev,
    #         max_elev=self.zero123plus_superres_max_elev,
    #         begin_rad=0,
    #         cam_weights=[3.0] + [1.0] * 5,
    #         reg_cam_weights=[0.5, 0.5],
    #         ref_img=in_img,
    #         ref_pose=ref_pose,
    #         ref_intrinsics=intrinsics,
    #         ref_intrinsics_size=intrinsics_size,
    #     )

    out_path = os.path.join(cache_dir, f"output_{uuid.uuid4()}.glb")
    out_mesh.write(out_path, flip_yz=True)
    print("Zero123++ to mesh finished.")

    pipe = None
    normal_model = matcher = vae = text_encoder = tokenizer = unet = scheduler = (
        image_enhancer
    ) = mesh_renderer = segmentation = nerf = tonemapping = controlnet = (
        controlnet_depth
    ) = None
    gc.collect()
    torch.cuda.empty_cache()

    return out_path


def run_video(
    proc_dict,
    front_view_id,
    distance,
    elevation,
    fov,
    length,
    resolution,
    lossless,
    layer="RGB",
    cache_dir=None,
    fps=30,
    render_bs=8,
):
    torch.cuda.empty_cache()
    proc_dict = json.loads(proc_dict)
    mesh_path = proc_dict["mesh_path"]
    in_mesh = Mesh.load(mesh_path, device=device, flip_yz=True)

    if front_view_id is not None and 0 <= front_view_id < self.preproc_num_views:
        front_azi = front_view_id / self.preproc_num_views * (2 * math.pi)
        print(f"\nUsing front view id {front_view_id}...")
    else:
        front_azi = 0

    elevation = np.radians(elevation)
    num_cameras = int(round(length * fps))
    camera_poses = random_surround_views(
        distance,
        num_cameras,
        elevation,
        elevation,
        use_linspace=True,
        begin_rad=front_azi,
    )[:, :3].to(device)
    focal = resolution / (2 * np.tan(np.radians(fov / 2)))
    intrinsics = torch.tensor(
        [focal, focal, resolution / 2, resolution / 2],
        dtype=torch.float32,
        device=device,
    )

    mesh_renderer = copy(self.mesh_renderer)
    mesh_renderer.ssaa = 2

    out_path = os.path.join(cache_dir, f"video_{uuid.uuid4()}.mp4")
    writer = VideoWriter(
        out_path, resolution=(resolution, resolution), lossless=lossless, fps=fps
    )
    for pose_batch in camera_poses.split(render_bs, dim=0):
        render_out = mesh_renderer(
            [in_mesh],
            pose_batch[None],
            intrinsics[None, None],
            resolution,
            resolution,
            normal_bg=[1.0, 1.0, 1.0],
            shading_fun=self.normal_shading_fun if in_mesh.textureless else None,
        )
        if layer == "RGB":
            image_batch = render_out["rgba"][0]
            image_batch = image_batch[..., :3] + (1 - image_batch[..., 3:4])
        elif layer == "Normal":
            image_batch = render_out["normal"][0]
        else:
            raise ValueError(f"Unknown layer: {layer}")
        image_batch = (
            torch.round(image_batch.clamp(min=0, max=1) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        for image_single in image_batch:
            writer.write(image_single)
    writer.close()

    return out_path


def process_v0(prompt):
    image_defaults["prompt"] = prompt

    # Text to image
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

    ## Inference
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

    image.convert("RGB").save("image2.jpg")

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
        mv_result = img2mv_pipeline(image, num_inference_steps=32).images[0]
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

    # Multi-view to 3D
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    mesh_path = run_zero123plus_to_mesh(
        "cuda:0",
        seed,
        image.convert("RGBA"),
        *nerf_mesh_defaults.values(),
        *{f"superres_{k}": v for k, v in superres_defaults.items()}.values(),
        *[np.asarray(Image.fromarray(img).convert("RGBA")) for img in mv_images],
        cache_dir=cache_dir,
    )
    print(mesh_path)
    return mesh_path

    # 3D to video


def process(prompt):
    image_defaults["prompt"] = prompt

    runner = MVEditRunner(
        device=torch.device("cuda"),
        local_files_only=False,
        unload_models=True,
        out_dir=None,
        save_interval=None,
        debug=False,
        no_safe=True,
    )

    # Text to image
    image = runner.run_text_to_img(seed, *image_defaults.values())
    # image.save("image1.png")

    # Segment first plan objects
    image = runner.run_segmentation(image)
    # image.save("image2.png")

    # Image to multi-view
    mv_images = runner.run_zero123plus(seed, image)
    # for i, img in enumerate(mv_images):
    #     img = Image.fromarray(img).convert("RGB")
    #     img.save(f"image3_{i}.jpg")

    # Multi-view to 3D
    cache_dir = Path("./meshes")
    cache_dir.mkdir(exist_ok=True, parents=True)
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

    # 3D to video


# Configs
seed = 42
device = "cuda:0"

# image_defaults["prompt"] = prompt
image_defaults["scheduler"] = "DPMSolverMultistep"
image_defaults["negative_prompt"] = ""
image_defaults["steps"] = 32
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
    # process(prompt)
    import pandas as pd
    from tqdm import tqdm

    data_path = "/home/and/projects/itmo/diploma/data/products/dataset_filtered_with_summary_v0.2.csv"
    df = pd.read_csv(data_path)
    prompts = df.product_summary.values

    # mesh_paths = ["" for _ in range(len(df))]
    # for i, prompt in enumerate(tqdm(prompts)):
    #     mesh_paths[i] = process(prompt)
    #     df["mesh_path"] = mesh_paths
    #     df.to_csv("data_with_meshes.csv", index=False)

    