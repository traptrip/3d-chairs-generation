# 3d-chairs-generation
Service of generation of 3D models of chairs 
Example of the service operation:
![](assets/chairs_gen_demo.gif)

# Start on host machine
## Install requirements
```
pip install -r requirements.txt -r requirements-cuda.txt
```

## Run gradio app 
```
python gradio_app.py --unload-models
```

# Run in Docker 
```
docker build -t 3d-chairs-gen .
docker run --gpus all -d -p 7860:7860 --rm --name 3d-chairs-service 3d-chairs-gen
```

# Stable Diffusion adapter training
- [Dataset](https://www.kaggle.com/datasets/traptrip/text-to-chair/data)
- [Kernel](https://www.kaggle.com/code/traptrip/stablediffusion-lora-text-to-chair)

# References
- [threestudio](https://github.com/threestudio-project/threestudio)
- [Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion)
- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus)
- [MVEdit](https://github.com/Lakonik/MVEdit)
- [TRACER](https://github.com/Karel911/TRACER)
- [LoFTR](https://github.com/zju3dv/LoFTR)
- [Omnidata](https://github.com/EPFL-VILAB/omnidata)
- [Imagepacker](https://github.com/theFroh/imagepacker)
