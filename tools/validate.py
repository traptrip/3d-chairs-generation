from pathlib import Path

import torch
import pandas as pd
import ImageReward as RM
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from lib.models.decoders.base_mesh_renderer import MeshRenderer
from lib.models.autoencoders.base_mesh import Mesh, preprocess_mesh
from lib.core.utils.camera_utils import random_surround_views

reward_model = RM.load("ImageReward-v1.0")
reward_model.eval()


def render_views_from_mesh(
    in_mesh_path,
    cache_dir,
    device,
    camera_distance=4,
    num_views=4,
    preproc_render_size=256,
    fov=30,
    render_bs=4,
):
    mesh_renderer = MeshRenderer(
        near=0.01, far=100, ssaa=1, texture_filter="linear-mipmap-linear"
    ).to(device)

    if not ".obj" in in_mesh_path:
        proc_dict = preprocess_mesh(in_mesh_path, cache_dir=cache_dir)
        if "mesh_obj" in proc_dict:
            in_mesh = proc_dict["mesh_obj"]
        else:
            in_mesh = Mesh.load(
                proc_dict["mesh_path"], device=device, auto_uv=True, flip_yz=True
            )
    else:
        in_mesh = Mesh.load(in_mesh_path, device=device, auto_uv=True, flip_yz=False)
    in_mesh = in_mesh.to(device)
    camera_poses = random_surround_views(
        camera_distance,
        num_views,
        0,
        0,
        use_linspace=True,
        begin_rad=0,
    )[:, :3].to(device)

    focal = preproc_render_size / (2 * np.tan(np.radians(fov / 2)))
    intrinsics = torch.tensor(
        [focal, focal, preproc_render_size / 2, preproc_render_size / 2],
        dtype=torch.float32,
        device=device,
    )

    mv_images = []
    for pose_batch in camera_poses.split(render_bs, dim=0):
        render_out = mesh_renderer(
            [in_mesh],
            pose_batch[None],
            intrinsics[None, None],
            preproc_render_size,
            preproc_render_size,
            shading_fun=None,
        )
        image_batch = render_out["rgba"][0]
        image_batch[..., :3] /= image_batch[..., 3:4].clamp(min=1e-6)
        image_batch = torch.round(image_batch * 255).to(torch.uint8).cpu().numpy()
        for image_single in image_batch:
            mv_images.append(Image.fromarray(image_single))

    return mv_images


@torch.inference_mode()
def validate(prompt: str, in_mesh_path=None, mv_images=None, **mesh_kwargs):
    if in_mesh_path and not mv_images:
        mv_images = render_views_from_mesh(in_mesh_path, **mesh_kwargs)
        mv_images = mv_images[:2] + mv_images[-2:]
        new_mv_images = []
        for img in mv_images:
            new_img = Image.new("RGBA", img.size, "WHITE")
            new_img.paste(img, (0, 0), img)
            new_mv_images.append(new_img.convert("RGB"))
        mv_images = new_mv_images

    rewards = reward_model.score(prompt, mv_images)
    return rewards, mv_images


if __name__ == "__main__":
    mesh_path_col = (
        "mesh_path"  # mesh_path dreamfusion-sd mvdream-sd21-rescale0.5 prolificdreamer
    )

    mesh_kwargs = dict(
        cache_dir=".",
        device="cuda:0",
        camera_distance=4,
        num_views=8,
        preproc_render_size=256,
        fov=30,
        render_bs=4,
    )
    df = pd.read_csv("./data/val.csv")
    df = df.dropna(subset=[mesh_path_col])
    prompts = df["product_summary"].values[:10]
    meshes = df[mesh_path_col].values
    all_rewards = []
    for prompt, mesh in tqdm(zip(prompts, meshes), total=len(prompts)):
        if mesh_path_col == "mesh_path" and mesh:
            rewards, mv_images = validate(prompt, mesh, **mesh_kwargs)

        elif mesh:
            images = list(list(Path(mesh).parents[1].glob("*-test"))[0].iterdir())
            mv_images = [
                Image.open(images[0]).crop((0, 0, 512, 512)).resize((256, 256)),
                Image.open(images[10]).crop((0, 0, 512, 512)).resize((256, 256)),
                Image.open(images[30]).crop((0, 0, 512, 512)).resize((256, 256)),
                Image.open(images[100]).crop((0, 0, 512, 512)).resize((256, 256)),
            ]
            rewards, _ = validate(prompt, mv_images=mv_images)

        print(mesh)
        print(rewards)
        for i, img in enumerate(mv_images):
            img.convert("RGB").save(f"{i}.jpg")
        all_rewards.extend(rewards)


all_rewards = np.array(all_rewards)
all_rewards = (all_rewards - all_rewards.min()) / (
    all_rewards.max() - all_rewards.min()
)
print(round(all_rewards.mean(), 4), round(all_rewards.std(), 4))
