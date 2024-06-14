import asyncio
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.auto import tqdm

DATA_DIR = Path("../data/products/images/")
# DATA_DIR.mkdir(parents=True, exist_ok=True)


async def download_image(session, url, filename, pbar):
    async with session.get(url) as response:
        if not Path(filename).parent.exists():
            Path(filename).parent.mkdir(parents=True)

        with open(filename, "wb") as f:
            while True:
                chunk = await response.content.read(1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


async def download_images(images_urls, images_paths):
    async with aiohttp.ClientSession() as session:
        tasks = []
        # with tqdm(total=len(df)) as pbar:
        for url, filepath in (
            pbar := tqdm(zip(images_urls, images_paths), total=len(images_urls))
        ):
            tasks.append(download_image(session, url, filepath, pbar))
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    df = pd.read_json("products_metadata.json")

    images_urls = [url for url in df["main_image_url"] if url] + [
        url for urls in df["other_images_urls"] for url in urls
    ]
    images_paths = [
        DATA_DIR / (url.split("/")[-1].split(".")[0] + ".jpg") for url in images_urls
    ]

    asyncio.run(download_images(images_urls, images_paths))
