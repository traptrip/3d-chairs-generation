import os
import time
import json
import traceback

import ikea_api
import yaml
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.common.exceptions import TimeoutException

from tqdm.auto import tqdm

# Constants like country, language, base url
constants = ikea_api.Constants(country="at", language="en")

# Initialize the browser
opt = Options()
opt.page_load_strategy = "eager"
opt.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# opt.headless = True
opt.add_argument("--headless")
driver_path = "/usr/local/bin/chromedriver"
# chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
# chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems


def process_product_page(url: str):
    driver = webdriver.Chrome(options=opt)

    res = {"url": url}

    # Load page
    driver.get(url)

    # time.sleep(0.1)

    # Load cluster
    navigation = driver.find_elements(
        By.XPATH, "//li[@class='bc-breadcrumb__list-item']"
    )
    res["cluster1"] = navigation[-2].text if len(navigation) > 1 else None
    res["cluster2"] = navigation[-3].text if len(navigation) > 2 else None
    res["cluster3"] = navigation[-4].text if len(navigation) > 3 else None

    # Get images urls
    # Extract the main image URL
    # images_grid = driver.find_element('div', class_='pip-media-grid__grid')
    try:
        res["main_image_url"] = (
            driver.find_element(By.XPATH, "//div[@data-type='MAIN_PRODUCT_IMAGE']")
            .find_element(By.XPATH, ".//img")
            .get_attribute("src")
        )
    except:
        res["main_image_url"] = ""

    try:
        res["other_images_urls"] = [
            el.find_element(By.XPATH, ".//img").get_attribute("src")
            for el in driver.find_elements(
                By.XPATH,
                "//div[@data-type='FUNCTIONAL_PRODUCT_IMAGE']",
            )
            + driver.find_elements(
                By.XPATH,
                "//div[@data-type='CONTEXT_PRODUCT_IMAGE']",
            )
        ]
    except:
        res["other_images_urls"] = []

    # Load product details
    try:
        wait(driver, 5).until(
            EC.element_to_be_clickable(
                (By.ID, "pip-product-information-section-list-0")
            )
        ).click()
    except TimeoutException:
        ...
    except:
        print(url)
        print(traceback.format_exc())

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Load short description
    try:
        res["product_header"] = soup.find(
            "span", class_="pip-header-section__description-text"
        ).get_text(strip=True)
    except:
        res["product_header"] = ""

    try:
        res["product_details"] = (
            soup.find("div", class_="pip-product-details")
            .find("div", class_="pip-product-details__container")
            .get_text(strip=True, separator="\n")
        )
    except:
        res["product_details"] = ""

    try:
        res["good_to_know_details"] = (
            soup.find("li", {"id": "HIDDEN_product-details-good-to-know"})
            .find("div", class_="pip-product-details__container")
            .get_text(strip=True, separator="\n")
        )
    except:
        res["good_to_know_details"] = ""

    try:
        res["materials_details"] = soup.find(
            "li", {"id": "HIDDEN_product-details-material-and-care"}
        ).get_text(strip=True, separator="\n")
    except:
        res["materials_details"] = ""

    driver.close()

    return res


def __get_ikea_product_pages(queries, limit=-1, saved_path="parsed_products.json"):
    # Search API
    search = ikea_api.Search(constants)
    # Search endpoint with prepared data
    for query in queries:
        print("Query:", query)
        parsed_products = []

        endpoint = search.search(query, limit=limit)
        results = ikea_api.run(endpoint)
        products = results["searchResultPage"]["products"]["main"]["items"]

        for p in tqdm(products):
            product = p["product"]
            product_data = {
                "product_id": product["id"],
                "product_url": product["pipUrl"],
                "main_image_url": product["mainImageUrl"],
                "main_image_description": product["mainImageAlt"],
                "colors": [c["name"] for c in product["colors"]],
                "homeFurnishingBusinessName": product["businessStructure"][
                    "homeFurnishingBusinessName"
                ],
                "productAreaName": product["businessStructure"]["productAreaName"],
                "productRangeAreaName": product["businessStructure"][
                    "productRangeAreaName"
                ],
                "categoryPath": [c["name"] for c in product["categoryPath"]],
            }

            parsed_products.append(product_data)

        if os.path.exists(saved_path):
            with open(saved_path, "r") as f:
                saved_products = json.load(f)
        else:
            saved_products = []

        with open(saved_path, "w") as f:
            json.dump(saved_products + parsed_products, f, ensure_ascii=False)


def get_ikea_product_pages_from_grid(driver: webdriver.Chrome, url: str):

    def check_products_grid(*args):
        return driver.find_elements(By.XPATH, "//div[@class='plp-fragment-wrapper']")

    # Load page
    driver.get(url)

    time.sleep(0.1)

    while True:

        bottom_info = driver.find_element(
            By.XPATH, "//progress[@class='plp-product-list__progress']"
        )

        # Do upload new products?
        cur_value = int(bottom_info.get_attribute("value"))
        max_products = int(bottom_info.get_attribute("max"))
        if cur_value < max_products:
            try:
                wait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, ".//span[@class='plp-btn__label']")
                    )
                ).click()
            except TimeoutException:
                break
            except:
                continue
        else:
            break

    try:
        product_cards = wait(driver, 5).until(check_products_grid)
    except TimeoutException:
        product_cards = []

    products_urls = [
        pc.find_element(By.XPATH, ".//a").get_attribute("href") for pc in product_cards
    ]

    print(f"{len(products_urls)} from {max_products} products parsed ({url})")

    return products_urls


if __name__ == "__main__":
    # driver = webdriver.Chrome(options=opt)
    products_urls_filepath = "products_urls.json"
    products_metadata_filepath = "products_metadata.json"

    # # Get products pages
    # if not os.path.exists(products_urls_filepath):
    #     with open("ikea_urls.yaml", "r") as f:
    #         urls = yaml.safe_load(f)["urls"]

    #     for url in tqdm(urls, desc="Scapping products"):
    #         products_urls = get_ikea_product_pages_from_grid(driver, url)

    #         # save new products
    #         if os.path.exists(products_urls_filepath):
    #             with open(products_urls_filepath, "r") as f:
    #                 saved_products = json.load(f)
    #         else:
    #             saved_products = []

    #         with open(products_urls_filepath, "w") as f:
    #             json.dump(saved_products + products_urls, f, ensure_ascii=False)

    # Get products metadata
    # if not os.path.exists(products_metadata_filepath):

    with open(products_metadata_filepath, "r") as f:
        products_metadata = json.load(f)

    with open(products_urls_filepath, "r") as f:
        products_urls = json.load(f)

    # products_metadata = []
    for url in tqdm(products_urls, desc="Processing products"):
        if url not in [p["url"] for p in products_metadata]:
            products_metadata.append(process_product_page(url))

            with open(products_metadata_filepath, "w") as f:
                json.dump(products_metadata, f)

    # driver.close()
