import concurrent.futures
import io
import os
import re
import time
from urllib.parse import urljoin, urlparse
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageSequence
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm.auto import tqdm


def parse_and_save_article(url: str, folder_path: str):
    """
    Parses an article from a given URL, extracts its title and content,
    downloads images, and saves them to a specified folder, converting
    GIF and WebP images to JPEG. For GIF images, it saves a few sample frames.

    Args:
        url (str): The URL of the article website.
        folder_path (str): The path to the folder where the article text
                            and images should be saved.
    """
    for _ in range(3):
        try:

            response = requests.get(url)
            if response.status_code == 429 or response.status_code == 504 or response.status_code == 500:
                time.sleep(5)
                continue
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            article_h1 = soup.find('h1')
            if not article_h1:
                return

            article_name = article_h1.get_text(strip=True)

            sanitized_article_name = re.sub(r'[\\/:"*?<>|]', '', article_name).strip()
            if not sanitized_article_name:
                sanitized_article_name = "untitled_article"

            article_content_div = soup.find('div', class_='prose--styled justify-self-center post_postContent__wGZtc')
            if not article_content_div:
                return

            article_text = ""

            for p_tag in article_content_div.find_all(['p', 'h2', 'h3', 'ul', 'ol']):
                article_text += p_tag.get_text(strip=True) + "\n\n"

            os.makedirs(folder_path, exist_ok=True)

            text_filename = os.path.join(folder_path, f"{sanitized_article_name}.txt")
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(article_name + "\n\n")
                f.write(article_text.strip())

            image_tags = article_content_div.find_all('img')
            for img_no, img_tag in enumerate(image_tags):
                img_src = img_tag.get('src')
                if img_src:

                    full_img_url = urljoin(url, img_src)

                    parsed_url = urlparse(full_img_url)
                    path_parts = parsed_url.path.split('/')
                    original_filename = path_parts[-1] if path_parts else f"image_{img_no}"

                    _, ext = os.path.splitext(original_filename)
                    ext = ext.lower()

                    try:
                        img_response = requests.get(full_img_url, stream=True)
                        img_response.raise_for_status()

                        image_data = io.BytesIO()
                        for chunk in img_response.iter_content(1024):
                            image_data.write(chunk)
                        image_data.seek(0)

                        if ext == '.gif':

                            try:
                                img = Image.open(image_data)
                                frame_count = 0
                                for i, frame in enumerate(ImageSequence.Iterator(img)):
                                    if frame_count >= 1:
                                        break
                                    frame_filename = os.path.join(folder_path,
                                                                  f"{sanitized_article_name}_{img_no}_frame_{i}.jpeg")
                                    frame.convert("RGB").save(frame_filename, "JPEG")
                                    frame_count += 1
                            except Exception as gif_e:
                                print(f"Error processing GIF {full_img_url}: {gif_e}")

                                with open(os.path.join(folder_path, f"{sanitized_article_name}_{img_no}.gif"),
                                          'wb') as img_file:
                                    img_file.write(image_data.getvalue())

                        elif ext == '.webp':

                            try:
                                img = Image.open(image_data)
                                image_filename = os.path.join(folder_path, f"{sanitized_article_name}_{img_no}.jpeg")
                                img.convert("RGB").save(image_filename, "JPEG")
                            except Exception as webp_e:

                                with open(os.path.join(folder_path, f"{sanitized_article_name}_{img_no}.webp"),
                                          'wb') as img_file:
                                    img_file.write(image_data.getvalue())
                        else:

                            image_filename = os.path.join(folder_path, f"{sanitized_article_name}_{img_no}{ext}")
                            with open(image_filename, 'wb') as img_file:
                                img_file.write(image_data.getvalue())

                    except requests.exceptions.RequestException as img_e:
                        print(f"Error downloading image {full_img_url}: {img_e}")
                    except Exception as general_img_e:
                        print(f"An error occurred while processing image {full_img_url}: {general_img_e}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            break


def get_all_articles_from_current_page(soup: BeautifulSoup, url: str) -> list[str]:
    """
    Helper function for getting all articles from current page.
    Args:
        soup: BeautifulSoup object to scrape the articles from
        url: link to page with articles

    Returns:
        list[str]: list of articles links
    """
    base_url = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
    soup = soup.find('div', class_="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-11")

    articles = soup.find_all('article')
    if not articles:
        print("No <article> elements found on the page.")
        return []

    article_urls = []
    for article in articles:

        link_tag = article.find_all('a')[-1]
        if link_tag and 'href' in link_tag.attrs:

            href = link_tag['href']
            if href.startswith('/'):

                article_urls.append(base_url + href)
            else:
                article_urls.append(href)

    return article_urls


def parse_articles_after_loading(url: str, category: str) -> list[str]:
    """
    Navigates to a URL, clicks a 'Load More' button until it's gone,
    and then scrapes the URLs of all <article> elements.

    Args:
        category: category name to save article to
        url: The URL of the target page.

    Returns:
        A list of URLs found within the href of an 'a' tag in each article.
        Returns an empty list if no articles are found or an error occurs.
    """

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        print(f"Successfully navigated to {url}")

        load_more_button_selector = (By.CSS_SELECTOR, "div.buttons_secondary__8o9u6.buttons_small__C_CFb.text-center")

        while True:
            try:

                load_more_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(load_more_button_selector)
                )

                driver.execute_script("arguments[0].click();", load_more_button)

                time.sleep(1)
            except (TimeoutException, NoSuchElementException):
                break
            except Exception as e:
                print(f"An unexpected error occurred while clicking the button: {e}")
                break

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

    return get_all_articles_from_current_page(soup, url), category


def parse_paged_articles(url: str, category: str) -> list[str]:
    """
    Navigates to a URL, clicks a 'Load More' button until it's gone,
    and then scrapes the URLs of all <article> elements.

    Args:
        category: category name to save article to
        url: The URL of the target page.

    Returns:
        A list of URLs found within the href of an 'a' tag in each article.
        Returns an empty list if no articles are found or an error occurs.
    """

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)

    article_urls = []
    try:
        driver.get(url)
        print(f"Successfully navigated to {url}")

        load_more_button_selector = (By.XPATH,
                                     "//div[normalize-space()='Older Posts' and contains(@class, 'buttons_secondary__8o9u6') and contains(@class, 'buttons_small__C_CFb') and contains(@class, 'text-center')]")

        while True:
            try:
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')

                article_urls += get_all_articles_from_current_page(soup, url)

                load_more_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(load_more_button_selector)
                )

                driver.execute_script("arguments[0].click();", load_more_button)

                time.sleep(1)
            except (TimeoutException, NoSuchElementException):
                break
            except Exception as e:
                print(f"An unexpected error occurred while clicking the button: {e}")
                break


    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

    return article_urls, category


def run_in_threads(tasks_to_process: list, max_workers: int = 8):
    """Helper function to ease using multithreading."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task[0], *task[1]) for task in tasks_to_process]

        results = []
        for future in tqdm(futures):
            results.append(future.result())

    return results


def get_complete_data(url: str = "https://www.deeplearning.ai/the-batch/"):
    """
    Main function to scrape the complete article pages. Saves all data in .data directory
    Args:
        url: url to the main the batch page

    Returns:
        None
    """
    base_url = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    categories = soup.find('ul', id='nav-secondary').find_all('li')

    category_urls = []

    for category in categories:
        c_url = category.find('a')['href']
        category_urls.append(base_url + c_url)

    download_categories_tasks = []

    for c_url in category_urls:
        response = requests.get(c_url)
        response.raise_for_status()
        cat = c_url.split('/')[-2]

        text = BeautifulSoup(response.text, 'html.parser').find('div',
                                                                class_='buttons_secondary__8o9u6 buttons_small__C_CFb text-center').text
        if text == 'Older Posts':
            download_categories_tasks.append([parse_paged_articles, [c_url, cat]])
        if text == 'Load More':
            download_categories_tasks.append([parse_articles_after_loading, [c_url, cat]])

    articles = []
    article_to_category = {}
    results = run_in_threads(download_categories_tasks, 9)
    for article_group, category in results:
        articles += article_group

        for article in article_group:
            article_to_category[article] = category

    articles = list(set(articles))
    base_folder = f'.data{os.sep}'
    dowload_articles_tasks = [
        (parse_and_save_article, [article, os.path.join(base_folder, article_to_category.get(article))]) for article in
        articles]
    d_results = run_in_threads(dowload_articles_tasks, 8)
    return None


if __name__ == '__main__':
    url = 'https://www.deeplearning.ai/the-batch/'
    get_complete_data()
