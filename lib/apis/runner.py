import gc
import os
import os.path as osp
import shutil
import math
import uuid
import json

from pathlib import Path
import numpy as np
import torch
import diffusers
import gradio as gr
from copy import copy
from functools import partial
from videoio import VideoWriter
from PIL import Image, ImageOps
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    StableDiffusionPipeline,
)
from segment_anything import sam_model_registry, SamPredictor
from mmcv.runner import set_random_seed
from mmcv.runner.checkpoint import _load_checkpoint
from huggingface_hub import hf_hub_download

from omnidata_modules.midas.dpt_depth import DPTDepthModel
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
from lib.core.webui.parameters import parse_3d_args, parse_2d_args
from lib.models.architecture.ip_adapter import IPAdapter
from lib.models.autoencoders.base_mesh import Mesh, preprocess_mesh
from lib.pipelines import (
    ThreeDPipeline,
    TextureSuperResPipeline,
    Zero123PlusPipeline,
)
from lib.pipelines.three_d_pipeline import default_max_num_views
from lib.pipelines.utils import (
    init_common_modules,
    rgba_to_rgb,
    do_segmentation,
    do_segmentation_pil,
    pad_rgba_image,
    join_prompts,
    zero123plus_postprocess,
)


def _api_wrapper(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        ret = func(*args, **kwargs)
        gc.collect()
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    return wrapper


class Runner:
    def __init__(
        self,
        device,
        local_files_only=False,
        empty_cache=True,
        unload_models=True,
        out_dir=None,
        save_interval=None,
        debug=False,
    ):
        self.local_files_only = local_files_only
        self.empty_cache = empty_cache
        self.unload_models = unload_models
        if out_dir is not None:
            self.out_dir_3d = osp.join(out_dir, "3d")
            self.out_dir_superres = osp.join(out_dir, "superres")
            self.out_dir_tex = osp.join(out_dir, "tex")
        else:
            self.out_dir_3d = self.out_dir_superres = self.out_dir_tex = None
        self.save_interval = save_interval
        self.debug = debug

        print("\nInitializing modules...")
        self.device = device
        (
            self.image_enhancer,
            self.mesh_renderer,
            self.segmentation,
            self.nerf,
            self.tonemapping,
            self.controlnet,
            self.controlnet_depth,
        ) = init_common_modules(self.device)

        self.stable_diffusion_checkpoint = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.unet = None
        self.safety_checker = None
        self.feature_extractor = None
        self.sd_adapter_ckpt = None

        self.controlnet_ip2p = None

        self.scheduler_ckpt = None
        self.scheduler_type = None
        self.scheduler = None

        self.ip_adapter = None
        self.ip_adapter_applied = False

        self.normal_model = None
        self.predictor = None
        self.matcher = None

        self.stablessdnerf = None

        self.zero123plus_pipe = None
        self.zero123plus_normal_pipe = None
        self.zero123plus_checkpoint = None

        self.zero123plus_pad_ratio = 0.75
        self.zero123plus1_2_pad_ratio = 0.9

        self.zero123plus_crop_ratio = 0.9
        self.zero123plus_crop_half_size = int(round(160 * self.zero123plus_crop_ratio))
        self.zero123plus_acutual_crop_ratio = self.zero123plus_crop_half_size / 160
        self.zero123plus_superres_camera_distance = 3.1
        self.zero123plus_superres_min_elev = 0.0
        self.zero123plus_superres_max_elev = 0.4
        self.zero123plus_superres_fov = 40
        self.zero123plus_superres_num_cameras = 6

        self.preproc_num_views = 12
        self.preproc_render_size = 256
        self.dummy_mv = [
            Image.new("RGBA", (self.preproc_render_size, self.preproc_render_size))
            for _ in range(self.preproc_num_views)
        ]

        self.proc_3d_to_3d_fov = 30
        self.proc_3d_to_3d_camera_distance = 3.7
        self.proc_3d_to_3d_min_elev = -0.3
        self.proc_3d_to_3d_max_elev = 0.6
        self.proc_3d_to_3d_tex_min_elev = -0.1
        self.proc_3d_to_3d_tex_max_elev = 0.3

        self.proc_retex_min_elev = -0.1
        self.proc_retex_max_elev = 0.5

        self.ssdnerf_camera_distance = 2.8
        self.ssdnerf_min_elev = 0.0
        self.ssdnerf_max_elev = 0.6
        self.ssdnerf_fov = 40
        self.ssdnerf_render_size = 160
        self.ssdnerf_front_azi = math.pi / 2
        self.ssdnerf_tex_min_elev = 0.0
        self.ssdnerf_tex_max_elev = 0.5

        print("Basic modules initialized.")

    def load_stable_diffusion(self, stable_diffusion_checkpoint, adapter_ckpt=None, adapter_filename=None):
        if stable_diffusion_checkpoint != self.stable_diffusion_checkpoint:
            print("\nLoading Stable Diffusion...")
            self.vae = AutoencoderKL.from_pretrained(
                stable_diffusion_checkpoint,
                subfolder="vae",
                torch_dtype=torch.float16,
                local_files_only=self.local_files_only,
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                stable_diffusion_checkpoint,
                subfolder="text_encoder",
                torch_dtype=torch.float16,
                local_files_only=self.local_files_only,
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                stable_diffusion_checkpoint,
                subfolder="tokenizer",
                torch_dtype=torch.float16,
                local_files_only=self.local_files_only,
            )
            self.unet = UNet2DConditionModel.from_pretrained(
                stable_diffusion_checkpoint,
                subfolder="unet",
                torch_dtype=torch.float16,
                local_files_only=self.local_files_only,
            )

            # LOAD LORA
            if adapter_ckpt:
                self.unet.load_attn_procs(adapter_ckpt, weight_name=adapter_filename)

            self.vae.to(self.device)
            self.text_encoder.to(self.device)
            self.unet.to(self.device)
            self.stable_diffusion_checkpoint = stable_diffusion_checkpoint
            self.ip_adapter_applied = False
            print("Stable Diffusion loaded.")
            gc.collect()

    def unload_stable_diffusion(self):
        self.stable_diffusion_checkpoint = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.unet = None
        self.safety_checker = None
        self.feature_extractor = None

        self.scheduler_ckpt = None
        self.scheduler_type = None
        self.scheduler = None

        gc.collect()
        torch.cuda.empty_cache()

    def load_controlnet_ip2p(self):
        if self.controlnet_ip2p is None:
            print("\nLoading InstructPix2Pix ControlNet...")
            self.controlnet_ip2p = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11e_sd15_ip2p",
                torch_dtype=torch.float16,
                local_files_only=self.local_files_only,
            ).to(self.device)
            print("InstructPix2Pix ControlNet loaded.")
            gc.collect()

    def unload_controlnet_ip2p(self):
        if self.controlnet_ip2p is not None:
            print("\nUnloading InstructPix2Pix ControlNet...")
            self.controlnet_ip2p = None
            print("InstructPix2Pix ControlNet unloaded.")
            gc.collect()
            torch.cuda.empty_cache()

    def load_scheduler(self, stable_diffusion_checkpoint, scheduler_type):
        print("\nLoading scheduler...")
        if scheduler_type.endswith("Karras"):
            extra_kwargs = dict(use_karras_sigmas=True, timestep_spacing="leading")
            scheduler_class = scheduler_type[:-6]
        else:
            extra_kwargs = dict(use_karras_sigmas=False, timestep_spacing="trailing")
            scheduler_class = scheduler_type
        sampler_class = getattr(diffusers.schedulers, scheduler_class + "Scheduler")
        if (
            stable_diffusion_checkpoint != self.scheduler_ckpt
            or scheduler_type != self.scheduler_type
        ):
            scheduler = sampler_class.from_pretrained(
                stable_diffusion_checkpoint,
                subfolder="scheduler",
                torch_dtype=torch.float16,
                local_files_only=self.local_files_only,
            )
            cfg = dict(scheduler.config)
            if "_use_default_values" in cfg:
                del cfg["_use_default_values"]
            cfg.update(extra_kwargs)
        else:
            cfg = dict(self.scheduler.config)
        self.scheduler = sampler_class.from_config(cfg)
        self.scheduler_ckpt = stable_diffusion_checkpoint
        self.scheduler_type = scheduler_type
        print("Scheduler loaded.")
        gc.collect()

    def load_ip_adapter(self, pipe):
        for module in [self.controlnet, self.controlnet_depth, self.controlnet_ip2p]:
            if module is not None:
                module.set_use_memory_efficient_attention_xformers(
                    not hasattr(torch.nn.functional, "scaled_dot_product_attention")
                )
        if not self.ip_adapter_applied:
            pipe.set_use_memory_efficient_attention_xformers(
                not hasattr(torch.nn.functional, "scaled_dot_product_attention")
            )
            print("\nLoading IP-Adapter...")
            self.ip_adapter = IPAdapter(
                pipe,
                "huggingface://h94/IP-Adapter/models/ip-adapter-plus_sd15.bin",
                "h94/IP-Adapter",
                local_files_only=self.local_files_only,
                device=self.device,
                dtype=torch.float16,
            )
            self.ip_adapter_applied = True
            print("IP-Adapter loaded.")
        gc.collect()

    def unload_ip_adapter(self, pipe):
        if self.ip_adapter_applied:
            print("\nUnloading IP adapter...")
            for module in [
                self.unet,
                self.controlnet,
                self.controlnet_depth,
                self.controlnet_ip2p,
            ]:
                if module is not None:
                    module.set_default_attn_processor()
            self.ip_adapter = None
            self.ip_adapter_applied = False
            print("IP adapter unloaded.")
        pipe.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        gc.collect()
        torch.cuda.empty_cache()

    def load_normal_model(self):
        if self.normal_model is None:
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
            normal_model.to(device=self.device, dtype=torch.float16)
            self.normal_model = normal_model
            print("Normal model loaded.")
            gc.collect()

    def unload_normal_model(self):
        if self.normal_model is not None:
            print("\nUnloading normal model...")
            self.normal_model = None
            print("Normal model unloaded.")
            gc.collect()
            torch.cuda.empty_cache()

    def load_sam_predictor(self):
        if self.predictor is None:
            print("\nLoading SAM...")
            ckpt_path = hf_hub_download(
                "ybelkada/segment-anything",
                "checkpoints/sam_vit_h_4b8939.pth",
                local_files_only=self.local_files_only,
            )
            sam = sam_model_registry["vit_h"](checkpoint=ckpt_path).to(
                device=self.device
            )
            self.predictor = SamPredictor(sam)
            print("SAM loaded.")
            gc.collect()

    def unload_sam_predictor(self):
        if self.predictor is not None:
            print("\nUnloading SAM...")
            self.predictor = None
            print("SAM unloaded.")
            gc.collect()
            torch.cuda.empty_cache()

    def load_zero123plus_pipeline(self, checkpoint, normal_controlnet=None):
        if checkpoint != self.zero123plus_checkpoint:
            print("\nLoading Zero123++...")
            zero123plus_pipe = Zero123PlusPipeline.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                local_files_only=self.local_files_only,
            )
            zero123plus_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                zero123plus_pipe.scheduler.config, timestep_spacing="trailing"
            )
            zero123plus_pipe.to(self.device)
            self.zero123plus_pipe = zero123plus_pipe
            self.zero123plus_checkpoint = checkpoint
            if normal_controlnet is not None:
                zero123plus_normal_pipeline = copy(zero123plus_pipe)
                zero123plus_normal_pipeline.add_controlnet(
                    ControlNetModel.from_pretrained(
                        normal_controlnet,
                        torch_dtype=torch.bfloat16,
                        local_files_only=self.local_files_only,
                    ),
                    conditioning_scale=1.0,
                )
                zero123plus_normal_pipeline.to(self.device)
                self.zero123plus_normal_pipe = zero123plus_normal_pipeline
            else:
                self.zero123plus_normal_pipe = None
            print("Zero123++ loaded.")
            gc.collect()

    def unload_zero123plus_pipeline(self):
        if self.zero123plus_pipe is not None:
            print("\nUnloading Zero123++...")
            self.zero123plus_pipe = None
            self.zero123plus_normal_pipe = None
            self.zero123plus_checkpoint = None
            print("Zero123++ unloaded.")
            gc.collect()
            torch.cuda.empty_cache()

    def load_matcher(self):
        if self.matcher is None:
            print("\nLoading feature matcher...")
            self.matcher = init_matcher().to(self.device)
            print("Feature matcher loaded.")
            gc.collect()

    def unload_matcher(self):
        if self.matcher is not None:
            print("\nUnloading feature matcher...")
            self.matcher = None
            print("Feature matcher unloaded.")
            gc.collect()
            torch.cuda.empty_cache()

    def normal_shading_fun(
        self, world_pos=None, albedo=None, world_normal=None, fg_mask=None, **kwargs
    ):
        assert albedo is not None and world_normal is not None
        shading = (world_normal[:, 2:] / 2 + 0.5).clamp(min=1e-6, max=1).sqrt()
        return self.tonemapping.lut(
            self.tonemapping.inverse_lut(albedo) + shading.log2()
        )

    def proc_zero123plus(
        self, seed, in_img, seg_padding, out_margin=0, num_inference_steps=40, pbar=None
    ):
        init_images = []
        init_normals = []
        set_random_seed(seed, deterministic=True)

        for _ in range(3):
            mv_result = self.zero123plus_pipe(
                in_img, num_inference_steps=num_inference_steps, guidance_scale=4.0
            ).images[0]
            if self.zero123plus_normal_pipe is not None:
                mv_normal = self.zero123plus_normal_pipe(
                    in_img,
                    depth_image=mv_result,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=4,
                ).images[0]
                mv_normal = (
                    np.asarray(mv_normal)
                    .reshape(3, 320, 2, 320, 3)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(6, 320, 320, 3)
                )
                for img in mv_normal:
                    init_normals.append(img)
            mv_result = (
                np.asarray(mv_result)
                .reshape(3, 320, 2, 320, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(6, 320, 320, 3)
            )
            for img in mv_result:
                init_images.append(Image.fromarray(img))
            if pbar:
                pbar.update(1)

            in_img_mirror = ImageOps.mirror(in_img)
            mv_result = self.zero123plus_pipe(
                in_img_mirror,
                num_inference_steps=num_inference_steps,
                guidance_scale=4.0,
            ).images[0]
            if self.zero123plus_normal_pipe is not None:
                mv_normal = self.zero123plus_normal_pipe(
                    in_img_mirror,
                    depth_image=mv_result,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=4,
                ).images[0]
                mv_normal = (
                    np.array(mv_normal)
                    .reshape(3, 320, 2, 320, 3)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(6, 320, 320, 3)
                )
                mv_normal[..., 0] = 255 - mv_normal[..., 0]
                for img in mv_normal:
                    init_normals.append(ImageOps.mirror(Image.fromarray(img)))
            mv_result = (
                np.asarray(mv_result)
                .reshape(3, 320, 2, 320, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(6, 320, 320, 3)
            )
            for img in mv_result:
                init_images.append(ImageOps.mirror(Image.fromarray(img)))
            if pbar:
                pbar.update(1)

        if self.zero123plus_normal_pipe is not None:
            init_images_rgba = do_segmentation_pil(
                init_images,
                self.segmentation,
                padding=seg_padding,
                bg_color=[0.5, 0.5, 0.5],
            )
            init_alphas = [np.asarray(img)[..., 3:] for img in init_images_rgba]
            init_images_ = []
            init_normals_ = []
            for img, normal, alpha in zip(init_images, init_normals, init_alphas):
                img, normal = zero123plus_postprocess(img, normal)
                img = np.array(img)
                img[..., 3:] = np.minimum(img[..., 3:], alpha)
                init_images_.append(Image.fromarray(img))
                init_normals_.append(normal)
            init_images = init_images_
            init_normals = init_normals_
        else:
            init_images = do_segmentation_pil(
                init_images,
                self.segmentation,
                padding=seg_padding,
                bg_color=[0.5, 0.5, 0.5],
            )
        init_images += init_normals
        if out_margin > 0:
            init_images = [
                np.asarray(image)[out_margin:-out_margin, out_margin:-out_margin]
                for image in init_images
            ]
        return init_images

    def proc_nerf_mesh(
        self,
        pipe,
        seed,
        nerf_mesh_kwargs,
        superres_kwargs,
        front_azi=None,
        camera_poses=None,
        use_reference=False,
        use_normal=False,
        pbar=None,
        **kwargs,
    ):
        print(nerf_mesh_kwargs)
        if self.out_dir_3d is not None:
            if os.path.exists(self.out_dir_3d):
                shutil.rmtree(self.out_dir_3d)
            os.makedirs(self.out_dir_3d)
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
            out_dir=self.out_dir_3d,
            save_interval=self.save_interval,
            save_all_interval=self.save_interval,
            mesh_reduction=128 / nerf_mesh_kwargs["tet_resolution"],
            max_num_views=partial(
                default_max_num_views,
                start_num=nerf_mesh_kwargs["max_num_views"],
                mid_num=nerf_mesh_kwargs["max_num_views"] // 2,
            ),
            debug=self.debug,
            pbar=pbar,
            **kwargs,
        )
        return out_mesh, ingp_states

    def proc_texture_superres(
        self,
        pipe,
        seed,
        in_model,
        ingp_states,
        base_kwargs,
        tex_superres_kwargs,
        camera_distance=None,
        fov=None,
        num_cameras=None,
        min_elev=None,
        max_elev=None,
        begin_rad=None,
        front_azi=None,
        ref_img=None,
        ref_pose=None,
        ref_intrinsics=None,
        ref_intrinsics_size=None,
        **kwargs,
    ):
        print(tex_superres_kwargs)
        if self.out_dir_superres is not None:
            if os.path.exists(self.out_dir_superres):
                shutil.rmtree(self.out_dir_superres)
            os.makedirs(self.out_dir_superres)
        set_random_seed(seed, deterministic=True)
        superres_camera_poses = random_surround_views(
            camera_distance,
            num_cameras,
            min_elev,
            max_elev,
            use_linspace=True,
            begin_rad=0 if begin_rad is None else begin_rad,
        )[:, :3].to(self.device)
        if ref_pose is not None:
            superres_camera_poses[0] = ref_pose
        superres_focal = 512 / (2 * np.tan(np.radians(fov / 2)))
        superres_intrinsics = (
            torch.tensor(
                [superres_focal, superres_focal, 256, 256],
                dtype=torch.float32,
                device=self.device,
            )[None, :]
            .expand(len(superres_camera_poses) + 2, -1)
            .clone()
        )
        if ref_intrinsics is not None and ref_intrinsics_size is not None:
            superres_intrinsics[0] = ref_intrinsics * (512 / ref_intrinsics_size)
        reg_camera_poses = [
            get_pose_from_angles_np(
                np.zeros((1,), dtype=np.float32),
                np.array([np.pi / 2], dtype=np.float32),
                np.array([camera_distance], dtype=np.float32),
            )[0, :3],
            get_pose_from_angles_np(
                np.zeros((1,), dtype=np.float32),
                np.array([-np.pi / 2], dtype=np.float32),
                np.array([camera_distance], dtype=np.float32),
            )[0, :3],
        ]
        prompts = (
            base_kwargs["prompt"]
            if front_azi is None
            else [
                join_prompts(base_kwargs["prompt"], view_prompt)
                for view_prompt in view_prompts(superres_camera_poses, front_azi)
            ]
        )
        out_mesh = pipe(
            prompt=prompts,
            negative_prompt=base_kwargs["negative_prompt"],
            in_model=in_model,
            ingp_states=ingp_states,
            cond_images=(
                [ref_img] * self.zero123plus_superres_num_cameras
                if ref_img is not None
                else None
            ),
            camera_poses=superres_camera_poses,
            reg_camera_poses=reg_camera_poses,
            intrinsics=superres_intrinsics,
            intrinsics_size=512,
            use_reference=ref_img is not None,
            guidance_scale=tex_superres_kwargs["cfg_scale"],
            num_inference_steps=tex_superres_kwargs["steps"],
            denoising_strength=(
                None
                if tex_superres_kwargs["random_init"]
                else tex_superres_kwargs["denoising_strength"]
            ),
            patch_size=tex_superres_kwargs["patch_size"],
            patch_bs=tex_superres_kwargs["patch_bs"],
            diff_bs=base_kwargs["diff_bs"],
            render_bs=base_kwargs["render_bs"],
            n_inverse_steps=tex_superres_kwargs["n_inverse_steps"],
            ip_adapter=self.ip_adapter,
            ip_adapter_use_cond_idx=[0] if ref_img is not None else None,
            lr_schedule=lambda p: tex_superres_kwargs["start_lr"]
            - (tex_superres_kwargs["start_lr"] - tex_superres_kwargs["end_lr"]) * p,
            default_prompt=tex_superres_kwargs["aux_prompt"],
            default_neg_prompt=tex_superres_kwargs["aux_negative_prompt"],
            prog_bar=gr.Progress().tqdm,
            out_dir=self.out_dir_superres,
            save_interval=self.save_interval,
            save_all_interval=self.save_interval,
            force_auto_uv=base_kwargs.get("force_auto_uv", False),
            debug=self.debug,
            **kwargs,
        )
        return out_mesh

    @_api_wrapper
    def run_mesh_preproc(self, in_mesh, *args, cache_dir=None, render_bs=8):
        if self.empty_cache:
            torch.cuda.empty_cache()
        if in_mesh is None or os.path.getsize(in_mesh) > 10000000:
            return gr.Gallery(value=self.dummy_mv, selected_index=None), None, None
        if len(args) > 0:
            front_view_id = args[0]
        else:
            front_view_id = None

        proc_dict = preprocess_mesh(in_mesh, cache_dir=cache_dir)
        if "mesh_obj" in proc_dict:
            in_mesh = proc_dict["mesh_obj"]
        else:
            in_mesh = Mesh.load(
                proc_dict["mesh_path"], device=self.device, flip_yz=True
            )
        in_mesh = in_mesh.to(self.device)
        camera_poses = random_surround_views(
            self.proc_3d_to_3d_camera_distance,
            self.preproc_num_views,
            0,
            0,
            use_linspace=True,
            begin_rad=0,
        )[:, :3].to(self.device)
        focal = self.preproc_render_size / (
            2 * np.tan(np.radians(self.proc_3d_to_3d_fov / 2))
        )
        intrinsics = torch.tensor(
            [focal, focal, self.preproc_render_size / 2, self.preproc_render_size / 2],
            dtype=torch.float32,
            device=self.device,
        )

        mv_images = []
        for pose_batch in camera_poses.split(render_bs, dim=0):
            render_out = self.mesh_renderer(
                [in_mesh],
                pose_batch[None],
                intrinsics[None, None],
                self.preproc_render_size,
                self.preproc_render_size,
                shading_fun=self.normal_shading_fun if in_mesh.textureless else None,
            )
            image_batch = render_out["rgba"][0]
            image_batch[..., :3] /= image_batch[..., 3:4].clamp(min=1e-6)
            image_batch = torch.round(image_batch * 255).to(torch.uint8).cpu().numpy()
            for image_single in image_batch:
                mv_images.append(Image.fromarray(image_single))
        proc_dict.pop("mesh_obj", None)
        return (
            gr.Gallery(value=mv_images, selected_index=front_view_id),
            json.dumps(proc_dict),
            front_view_id,
        )

    @_api_wrapper
    def run_mesh_to_video(
        self,
        in_mesh_path,
        front_view_id=None,
        layer="RGB",
        distance=4,
        length=5,
        elevation=10,
        fov=30,
        resolution=512,
        fps=30,
        render_bs=8,
        cache_dir=None,
    ):
        # Get images
        if self.empty_cache:
            torch.cuda.empty_cache()

        proc_dict = preprocess_mesh(in_mesh_path, cache_dir=cache_dir)
        if "mesh_obj" in proc_dict:
            in_mesh = proc_dict["mesh_obj"]
        else:
            in_mesh = Mesh.load(
                proc_dict["mesh_path"], device=self.device, flip_yz=True
            )
        in_mesh = in_mesh.to(self.device)
        camera_poses = random_surround_views(
            self.proc_3d_to_3d_camera_distance,
            self.preproc_num_views,
            0,
            0,
            use_linspace=True,
            begin_rad=0,
        )[:, :3].to(self.device)
        focal = self.preproc_render_size / (
            2 * np.tan(np.radians(self.proc_3d_to_3d_fov / 2))
        )
        intrinsics = torch.tensor(
            [focal, focal, self.preproc_render_size / 2, self.preproc_render_size / 2],
            dtype=torch.float32,
            device=self.device,
        )

        mv_images = []
        for pose_batch in camera_poses.split(render_bs, dim=0):
            render_out = self.mesh_renderer(
                [in_mesh],
                pose_batch[None],
                intrinsics[None, None],
                self.preproc_render_size,
                self.preproc_render_size,
                shading_fun=self.normal_shading_fun if in_mesh.textureless else None,
            )
            image_batch = render_out["rgba"][0]
            image_batch[..., :3] /= image_batch[..., 3:4].clamp(min=1e-6)
            image_batch = torch.round(image_batch * 255).to(torch.uint8).cpu().numpy()
            for image_single in image_batch:
                mv_images.append(Image.fromarray(image_single))
        proc_dict.pop("mesh_obj", None)

        # Render video
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
        )[:, :3].to(self.device)
        focal = resolution / (2 * np.tan(np.radians(fov / 2)))
        intrinsics = torch.tensor(
            [focal, focal, resolution / 2, resolution / 2],
            dtype=torch.float32,
            device=self.device,
        )

        mesh_renderer = copy(self.mesh_renderer)
        mesh_renderer.ssaa = 2

        mesh_filename = Path(in_mesh_path).stem
        out_path = osp.join(cache_dir, f"video_{mesh_filename}.mp4")
        writer = VideoWriter(out_path, resolution=(resolution, resolution), fps=fps)
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

    @_api_wrapper
    def run_segmentation(self, in_img):
        self.load_sam_predictor()
        in_img_np = np.asarray(in_img)
        if in_img_np.shape[-1] == 4 and np.any(in_img_np[..., 3] != 255):
            in_img = in_img_np
        else:
            in_img = do_segmentation(
                in_img_np[None, :, :, :3],
                self.segmentation,
                sam_predictor=self.predictor,
                to_np=True,
            )[0]
        if self.unload_models:
            self.unload_sam_predictor()
            self.unload_controlnet_ip2p()
        if self.empty_cache:
            torch.cuda.empty_cache()
        return Image.fromarray(in_img)

    @_api_wrapper
    def run_zero123plus(self, seed, in_img, pbar=None):
        self.load_zero123plus_pipeline("sudo-ai/zero123plus-v1.1")
        in_img = pad_rgba_image(np.asarray(in_img), ratio=self.zero123plus_pad_ratio)
        print(f"\nRunning Zero123++ generation with seed {seed}...")
        init_images = self.proc_zero123plus(
            seed,
            in_img,
            seg_padding=32,
            out_margin=160 - self.zero123plus_crop_half_size,
            pbar=pbar,
        )
        print("Zero123++ generation finished.")
        if self.unload_models:
            self.unload_zero123plus_pipeline()
            self.unload_controlnet_ip2p()
        if self.empty_cache:
            torch.cuda.empty_cache()
        return init_images

    @_api_wrapper
    def run_zero123plus1_2(self, seed, in_img, pbar):
        self.load_zero123plus_pipeline(
            "sudo-ai/zero123plus-v1.2",
            normal_controlnet="sudo-ai/controlnet-zp12-normal-gen-v1",
        )
        in_img = pad_rgba_image(np.asarray(in_img), ratio=self.zero123plus1_2_pad_ratio)
        print(f"\nRunning Zero123++ generation with seed {seed}...")
        init_images = self.proc_zero123plus(seed, in_img, seg_padding=64, pbar=pbar)
        print("Zero123++ generation finished.")
        if self.unload_models:
            self.unload_zero123plus_pipeline()
            self.unload_controlnet_ip2p()
        if self.empty_cache:
            torch.cuda.empty_cache()
        return init_images

    @_api_wrapper
    def run_zero123plus_to_mesh(
        self, seed, in_img, *args, cache_dir=None, pbar=None, **kwargs
    ):
        nerf_mesh_kwargs, superres_kwargs, init_images = parse_3d_args(
            list(args), kwargs
        )

        self.load_normal_model()
        self.load_matcher()
        self.load_stable_diffusion(nerf_mesh_kwargs["checkpoint"])
        self.load_scheduler(
            nerf_mesh_kwargs["checkpoint"], nerf_mesh_kwargs["scheduler"]
        )
        pipe = ThreeDPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=[self.controlnet, self.controlnet_depth],
            scheduler=self.scheduler,
            nerf=self.nerf,
            mesh_renderer=self.mesh_renderer,
            image_enhancer=self.image_enhancer,
            segmentation=self.segmentation,
            normal_model=self.normal_model,
            tonemapping=self.tonemapping,
        )
        self.load_ip_adapter(pipe)
        print(f"\nRunning Zero123++ to mesh with seed {seed}...")

        in_img = pad_rgba_image(
            np.asarray(np.asarray(in_img)),
            ratio=self.zero123plus_pad_ratio / self.zero123plus_acutual_crop_ratio,
        )
        focal = 350
        fov = np.rad2deg(np.arctan(self.zero123plus_crop_half_size / focal) * 2)
        camera_distance = 1 / np.sin(np.radians(fov / 2))
        azims = [30, 90, 150, 210, 270, 330, 330, 270, 210, 150, 90, 30] * 3
        elevs = [30, -20] * 18
        camera_poses = get_pose_from_angles(
            torch.tensor(azims, dtype=torch.float32) * np.pi / 180,
            torch.tensor(elevs, dtype=torch.float32) * np.pi / 180,
            camera_distance,
        )[:, :3].to(self.device)
        intrinsics = torch.tensor(
            [
                focal,
                focal,
                self.zero123plus_crop_half_size,
                self.zero123plus_crop_half_size,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        intrinsics_size = self.zero123plus_crop_half_size * 2

        elev, in_pose = elev_estimation(
            Image.fromarray(rgba_to_rgb(in_img, bg_color=(127, 127, 127))),
            init_images,  # [init_images[i] for i in [0, 5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 35]],
            camera_poses,  # camera_poses[[0, 5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 35]],
            intrinsics,
            intrinsics_size,
            self.matcher,
        )
        init_images = [in_img] + list(init_images)
        camera_poses = torch.cat([in_pose[None, :3], camera_poses], dim=0)

        out_mesh, ingp_states = self.proc_nerf_mesh(
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
            ip_adapter=self.ip_adapter,
            use_reference=True,
            use_normal=True,
            pbar=pbar,
        )

        if superres_kwargs["do_superres"]:
            self.load_stable_diffusion(superres_kwargs["checkpoint"])
            self.load_scheduler(
                superres_kwargs["checkpoint"], superres_kwargs["scheduler"]
            )
            pipe = TextureSuperResPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                controlnet=[self.controlnet, self.controlnet_depth],
                scheduler=self.scheduler,
                nerf=self.nerf,
                mesh_renderer=self.mesh_renderer,
            )
            ref_pose = get_pose_from_angles(
                torch.zeros((1,), dtype=torch.float32, device=self.device),
                in_pose.new_tensor([(elev + 10) * np.pi / 180]),
                in_pose.new_tensor([camera_distance]),
            )[0, :3]
            if self.empty_cache:
                torch.cuda.empty_cache()
            out_mesh = self.proc_texture_superres(
                pipe,
                seed + 1 if seed < 2**31 else 0,
                out_mesh,
                ingp_states,
                nerf_mesh_kwargs,
                superres_kwargs,
                camera_distance=self.zero123plus_superres_camera_distance,
                fov=self.zero123plus_superres_fov,
                num_cameras=6,
                min_elev=self.zero123plus_superres_min_elev,
                max_elev=self.zero123plus_superres_max_elev,
                begin_rad=0,
                cam_weights=[3.0] + [1.0] * 5,
                reg_cam_weights=[0.5, 0.5],
                ref_img=in_img,
                ref_pose=ref_pose,
                ref_intrinsics=intrinsics,
                ref_intrinsics_size=intrinsics_size,
            )

        out_path = osp.join(cache_dir, f"output_{uuid.uuid4()}.glb")
        out_mesh.write(out_path, flip_yz=True)
        print("Zero123++ to mesh finished.")

        if self.unload_models:
            self.unload_normal_model()
            self.unload_matcher()
            self.unload_stable_diffusion()
            self.unload_sam_predictor()
            self.unload_controlnet_ip2p()
        if self.empty_cache:
            torch.cuda.empty_cache()

        return out_path

    @_api_wrapper
    def run_zero123plus1_2_to_mesh(
        self, seed, in_img, *args, cache_dir=None, pbar=None, **kwargs
    ):
        nerf_mesh_kwargs, superres_kwargs, init_images = parse_3d_args(
            list(args), kwargs
        )
        init_images, init_normals = (
            init_images[: len(init_images) // 2],
            init_images[len(init_images) // 2 :],
        )

        self.load_normal_model()
        self.load_matcher()
        self.load_stable_diffusion(nerf_mesh_kwargs["checkpoint"])
        self.load_scheduler(
            nerf_mesh_kwargs["checkpoint"], nerf_mesh_kwargs["scheduler"]
        )
        pipe = ThreeDPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=[self.controlnet, self.controlnet_depth],
            scheduler=self.scheduler,
            nerf=self.nerf,
            mesh_renderer=self.mesh_renderer,
            image_enhancer=self.image_enhancer,
            segmentation=self.segmentation,
            normal_model=self.normal_model,
            tonemapping=self.tonemapping,
        )
        self.load_ip_adapter(pipe)
        print(f"\nRunning Zero123++ to mesh with seed {seed}...")

        in_img = pad_rgba_image(np.asarray(np.asarray(in_img)), ratio=0.9)
        fov = 30
        focal = 320 / (2 * np.tan(np.radians(fov / 2)))
        camera_distance = 1 / np.sin(np.radians(fov / 2))
        azims = [30, 90, 150, 210, 270, 330, 330, 270, 210, 150, 90, 30] * 3
        elevs = [20, -10] * 18
        camera_poses = get_pose_from_angles(
            torch.tensor(azims, dtype=torch.float32) * np.pi / 180,
            torch.tensor(elevs, dtype=torch.float32) * np.pi / 180,
            camera_distance,
        )[:, :3].to(self.device)
        intrinsics = torch.tensor(
            [focal, focal, 160, 160], dtype=torch.float32, device=self.device
        )
        intrinsics_size = 320

        in_pose, elev, distance, focal, cx, cy = pose5dof_estimation(
            Image.fromarray(rgba_to_rgb(in_img, bg_color=(127, 127, 127))),
            [init_images[i] for i in [0, 5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 35]],
            camera_poses[[0, 5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 35]],
            intrinsics,
            intrinsics_size,
            self.matcher,
        )
        in_intrinsics = in_pose.new_tensor([focal, focal, cx, cy])
        init_images = [in_img] + list(init_images)
        init_normals = [None] + list(init_normals)
        intrinsics = torch.cat(
            [
                in_intrinsics[None, :],
                intrinsics[None, :].expand(camera_poses.size(0), -1),
            ],
            dim=0,
        )
        camera_poses = torch.cat([in_pose[None, :3], camera_poses], dim=0)

        out_mesh, ingp_states = self.proc_nerf_mesh(
            pipe,
            seed,
            nerf_mesh_kwargs,
            superres_kwargs,
            init_images=init_images,
            normals=init_normals,
            camera_poses=camera_poses,
            intrinsics=intrinsics,
            intrinsics_size=intrinsics_size,
            cam_weights=[2.0] + [1.1, 0.95, 0.9, 0.85, 1.0, 1.05] * 6,
            seg_padding=96,
            keep_views=[0],
            ip_adapter=self.ip_adapter,
            use_reference=True,
            use_normal=True,
            pbar=pbar,
        )

        if superres_kwargs["do_superres"]:
            self.load_stable_diffusion(superres_kwargs["checkpoint"])
            self.load_scheduler(
                superres_kwargs["checkpoint"], superres_kwargs["scheduler"]
            )
            pipe = TextureSuperResPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                controlnet=[self.controlnet, self.controlnet_depth],
                scheduler=self.scheduler,
                nerf=self.nerf,
                mesh_renderer=self.mesh_renderer,
            )
            ref_pose = get_pose_from_angles(
                torch.zeros((1,), dtype=torch.float32, device=self.device),
                in_pose.new_tensor([(elev + 10) * np.pi / 180]),
                in_pose.new_tensor([distance]),
            )[0, :3]
            if self.empty_cache:
                torch.cuda.empty_cache()
            out_mesh = self.proc_texture_superres(
                pipe,
                seed + 1 if seed < 2**31 else 0,
                out_mesh,
                ingp_states,
                nerf_mesh_kwargs,
                superres_kwargs,
                camera_distance=self.zero123plus_superres_camera_distance,
                fov=self.zero123plus_superres_fov,
                num_cameras=6,
                min_elev=self.zero123plus_superres_min_elev,
                max_elev=self.zero123plus_superres_max_elev,
                begin_rad=0,
                cam_weights=[3.0] + [1.0] * 5,
                reg_cam_weights=[0.5, 0.5],
                ref_img=in_img,
                ref_pose=ref_pose,
                ref_intrinsics=in_intrinsics,
                ref_intrinsics_size=intrinsics_size,
            )

        out_path = osp.join(cache_dir, f"output_{uuid.uuid4()}.glb")
        out_mesh.write(out_path, flip_yz=True)
        print("Zero123++ to mesh finished.")

        if self.unload_models:
            self.unload_normal_model()
            self.unload_matcher()
            self.unload_stable_diffusion()
            self.unload_sam_predictor()
            self.unload_controlnet_ip2p()
        if self.empty_cache:
            torch.cuda.empty_cache()

        return out_path

    @_api_wrapper
    def run_text_to_img(self, seed, *args, **kwargs):
        image_kwargs = parse_2d_args(list(args), kwargs)

        self.load_stable_diffusion(
            image_kwargs["checkpoint"],
            image_kwargs["adapter_ckpt"],
            image_kwargs["adapter_filename"],
        )
        self.load_scheduler(image_kwargs["checkpoint"], image_kwargs["scheduler"])
        pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=False,
        )
        self.unload_ip_adapter(pipe)

        print(f"\nRunning text-to-image with seed {seed}...")
        set_random_seed(seed, deterministic=True)
        out_img = pipe(
            height=image_kwargs["height"],
            width=image_kwargs["width"],
            prompt=join_prompts(image_kwargs["prompt"], image_kwargs["aux_prompt"]),
            negative_prompt=join_prompts(
                image_kwargs["negative_prompt"], image_kwargs["aux_negative_prompt"]
            ),
            num_inference_steps=image_kwargs["steps"],
            guidance_scale=image_kwargs["cfg_scale"],
            return_dict=False,
        )[0][0]
        print("Text-to-Image finished.")
        if self.unload_models:
            self.unload_stable_diffusion()
            self.unload_controlnet_ip2p()
        if self.empty_cache:
            torch.cuda.empty_cache()
        return out_img

    @_api_wrapper
    def run_video(
        self,
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
        if self.empty_cache:
            torch.cuda.empty_cache()
        proc_dict = json.loads(proc_dict)
        mesh_path = proc_dict["mesh_path"]
        in_mesh = Mesh.load(mesh_path, device=self.device, flip_yz=True)

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
        )[:, :3].to(self.device)
        focal = resolution / (2 * np.tan(np.radians(fov / 2)))
        intrinsics = torch.tensor(
            [focal, focal, resolution / 2, resolution / 2],
            dtype=torch.float32,
            device=self.device,
        )

        mesh_renderer = copy(self.mesh_renderer)
        mesh_renderer.ssaa = 2

        out_path = osp.join(cache_dir, f"video_{uuid.uuid4()}.mp4")
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
