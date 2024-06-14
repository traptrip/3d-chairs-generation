from pathlib import Path
import pandas as pd


MODEL_NAME = "prolificdreamer"  # "mvdream-sd21-rescale0.5"
MODEL_DIR = Path(
    f"/home/and/projects/itmo/diploma/libs/threestudio/outputs/{MODEL_NAME}"
)

df = pd.read_csv("val.csv")

mesh_paths = [""] * len(df)
exp_dirs = list(MODEL_DIR.iterdir())


def sort_key(p):
    try:
        return int(p.name.split("_")[0])
    except:
        return float("inf")


exp_dirs = sorted(exp_dirs, key=lambda p: sort_key(p))
exp_dirs = [d for d in exp_dirs if sort_key(d) != float("inf")]
for i, d in enumerate(exp_dirs):
    print(d)
    idx = int(d.name.split("_")[0]) - 1
    mesh_paths[idx] = list(d.rglob("*-export"))[0] / "model.obj"

df[MODEL_NAME] = mesh_paths
df.to_csv("val.csv", index=False)
