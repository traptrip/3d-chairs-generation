import os
import sys
from functools import partial
import shutil
import os.path as osp
import argparse

sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "16"

import torch
import gradio as gr

from lib.core.mvedit_webui.shared_opts import send_to_click
from lib.core.mvedit_webui.tab_img_to_3d import create_interface_img_to_3d
from lib.core.mvedit_webui.tab_text_to_img_to_3d import (
    create_interface_text_to_img_to_3d,
)
from lib.core.mvedit_webui.tab_3d_to_video import create_interface_3d_to_video
from lib.apis.mvedit import MVEditRunner


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

    with gr.Blocks(
        analytics_enabled=False,
        title="3D scene generation",
        css="lib/web/style.css",
    ) as demo:
        md_txt = "# 3D Chairs generation toolbox"
        if not args.advanced:
            md_txt += "<br>**Advanced settings** are disabled. Deploy the app with `--advanced` to enable them."
        gr.Markdown(md_txt)

        with gr.Tabs(selected="tab_img_to_3d") as main_tabs:
            with gr.TabItem("Text-to-Image-to-3D", id="tab_text_to_img_to_3d"):
                _, var_text_to_img_to_3d = create_interface_text_to_img_to_3d(
                    runner.run_text_to_img,
                    examples=[
                        [768, 512, "a wooden carving of a wise old turtle", ""],
                        [512, 512, "a glowing robotic unicorn, full body", ""],
                        [512, 512, "a ceramic mug shaped like a smiling cat", ""],
                    ],
                    advanced=args.advanced,
                )
            with gr.TabItem("Image-to-3D", id="tab_img_to_3d"):
                _, var_img_to_3d_1_1 = create_interface_img_to_3d(
                    runner.run_segmentation,
                    runner.run_zero123plus,
                    runner.run_zero123plus_to_mesh,
                    api_names=[
                        "image_segmentation",
                        "img_to_3d_1_1_zero123plus",
                        "img_to_3d_1_1_zero123plus_to_mesh",
                    ],
                    init_inverse_steps=640,
                    n_inverse_steps=80,
                    diff_bs=args.diff_bs,
                    advanced=args.advanced,
                )
            with gr.TabItem("Tools", id="tab_tools"):
                with gr.Tabs() as sub_tabs_tools:
                    with gr.TabItem("Export Video", id="tab_export_video"):
                        _, var_3d_to_video = create_interface_3d_to_video(
                            runner.run_mesh_preproc,
                            runner.run_video,
                            api_names=[False, "3d_to_video"],
                        )

        for var_dict in [var_img_to_3d_1_1]:
            instruct = var_dict.get("instruct", False)
            in_fields = (
                ["output"] if instruct else ["output", "prompt", "negative_prompt"]
            )
            out_fields = (
                ["in_mesh"] if instruct else ["in_mesh", "prompt", "negative_prompt"]
            )

            if "export_video" in var_dict:
                var_dict["export_video"].click(
                    fn=partial(
                        send_to_click, target_tab_ids=["tab_tools", "tab_export_video"]
                    ),
                    inputs=var_dict["output"],
                    outputs=[var_3d_to_video["in_mesh"]] + [main_tabs, sub_tabs_tools],
                    show_progress=False,
                    api_name=False,
                ).success(**var_3d_to_video["preproc_kwargs"], api_name=False)

        for i, var_img_to_3d in enumerate([var_img_to_3d_1_1]):
            var_text_to_img_to_3d[f"to_zero123plus1_{i + 1}"].click(
                fn=partial(
                    send_to_click,
                    target_tab_ids=["tab_img_to_3d", f"tab_zero123plus1_{i + 1}"],
                ),
                inputs=[
                    var_text_to_img_to_3d[k]
                    for k in ["output_image", "prompt", "negative_prompt"]
                ],
                outputs=[
                    var_img_to_3d[k] for k in ["in_image", "prompt", "negative_prompt"]
                ]
                + [main_tabs],
                show_progress=False,
                api_name=False,
            ).success(
                fn=lambda: gr.Accordion(open=True),
                inputs=None,
                outputs=var_img_to_3d["prompt_accordion"],
                show_progress=False,
                api_name=False,
            )

        demo.queue().launch(share=args.share, debug=args.debug)


if __name__ == "__main__":
    main()
