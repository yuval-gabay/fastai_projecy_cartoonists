import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.chrome.options import Options  # <-- NEW IMPORT

# --- SENSITIVE CONFIGURATION (FILL THESE IN YOURSELF) ---
PINTEREST_EMAIL = "gabayuv@gmail.com"  # <-- FILL YOUR EMAIL HERE
PINTEREST_PASSWORD = "Yuval142001"  # <-- FILL YOUR PASSWORD HERE
# --------------------------------------------------------

# --- CONFIGURATION: Define All Artists and Their Sources ---
# Note: Multiple URLs are listed for Timm, but the loop only picks the first URL for each 'name' entry.
# You can run the script multiple times, switching the URLs, or add them as separate list items.
ARTIST_SOURCES = [
    {
        'name': 'tartakovsky',
        'url': 'https://characterdesignreferences.com/art-of-animation-2/star-wars-clone-wars',  # New URL
        'dir': r'C:\Users\Surface\PycharmProjects\PythonProject2\imageData\tartakovsky'
    },
    {
        'name': 'timm',
        'url': 'https://characterdesignreferences.com/art-of-animation-9/art-of-batman-beyond?rq=justice',
        # New URL 1 (Design Ref)
        'dir': r'C:\Users\Surface\PycharmProjects\PythonProject2\imageData\timm'
    },
    # Secondary URL for Timm: Run this separately or change the 'timm' URL above.
    # {
    #     'name': 'timm',
    #     'url': 'https://characterdesignreferences.com/art-of-animation-4/art-of-batman-the-animated-series',
    #     'dir': r'C:\Users\Surface\PycharmProjects\PythonProject2\imageData\timm'
    # },
    {
        'name': 'pendleton',
        'url': 'https://www.pinterest.com/ideas/adventure-time-art-style-reference/955353713802/',  # New URL
        'dir': r'C:\Users\Surface\PycharmProjects\PythonProject2\imageData\pendleton'
    },
]

SCROLL_COUNT = 15
SCROLL_PAUSE_TIME = 2
TARGET_LIMIT = 180


# --- AUTHENTICATION FUNCTION ---
def authenticate_driver(driver):
    """Navigates to the Pinterest login page and attempts to log in."""
    if not PINTEREST_EMAIL or not PINTEREST_PASSWORD:
        return False

    print("Attempting Pinterest login...")
    driver.get("https://www.pinterest.com/login/")
    time.sleep(2)

    try:
        email_field = driver.find_element(By.NAME, 'id')
        email_field.send_keys(PINTEREST_EMAIL)

        password_field = driver.find_element(By.NAME, 'password')
        password_field.send_keys(PINTEREST_PASSWORD)

        login_button = driver.find_element(By.XPATH, "//button[@data-test-id='login-button']")
        login_button.click()

        # Wait for the login redirect
        WebDriverWait(driver, 10).until(EC.url_changes(driver.current_url))
        print("Login successful!")
        return True

    except Exception:
        print("Login failed or elements not found.")
        return False


# --- CORE SCRAPING FUNCTION ---
def scrape_images(source_info):
    """Handles browser setup, scrolling, image extraction, and saving."""

    url = source_info['url']
    download_dir = source_info['dir']
    artist_name = source_info['name']

    Path(download_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n--- Starting {artist_name.upper()} from {url} ---")

    # 1. SETUP STEALTH DRIVER
    print("Setting up stealthy Chrome WebDriver...")
    options = Options()
    # Hides the most common fingerprint (the 'webdriver' property)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    # Adds a human User-Agent
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)  # Pass the options
    except Exception as e:
        print(f"ERROR: Could not set up WebDriver. {e}")
        return

    # 2. AUTHENTICATION (Only for Pinterest links)
    is_pinterest = 'pinterest' in url
    if is_pinterest:
        # Attempt to login only if necessary, which is what Pinterest requires
        if authenticate_driver(driver):
            print(f"Navigating to target URL: {url}")
            driver.get(url)
        else:
            print("Skipping login; navigating directly to target (may be blocked).")
            driver.get(url)  # Proceed anyway
    else:
        # For character design reference sites, just navigate
        driver.get(url)

    image_urls = set()

    # 3. Scroll the page to load pins/images
    last_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(SCROLL_COUNT):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height and i > 2:
            print("Reached end of page or no new content loaded.")
            break
        last_height = new_height

    # 4. Extract all image URLs
    img_elements = driver.find_elements(By.TAG_NAME, 'img')

    for img in img_elements:
        try:
            src = img.get_attribute('src') or img.get_attribute('data-src')

            if src and 'http' in src and ('200x' not in src and 'icon' not in src):
                if 'pinterest' in url:
                    high_res_src = src.replace('/236x', '/474x')
                else:
                    high_res_src = src
                image_urls.add(high_res_src)
        except Exception:
            continue

    driver.quit()
    print(f"Found {len(image_urls)} unique image URLs.")

    # 5. Download Images
    downloaded_count = 0
    for i, url in enumerate(list(image_urls)):
        if downloaded_count >= TARGET_LIMIT:
            print(f"Reached target of {TARGET_LIMIT} images. Stopping download.")
            break

        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            file_extension = os.path.splitext(url.split('/')[-1])[1]
            if not file_extension or len(file_extension) > 5:
                file_extension = '.jpg'

            file_name = f"pin_{artist_name}_{i + 1:03d}{file_extension}"
            file_path = os.path.join(download_dir, file_name)

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            downloaded_count += 1
            if downloaded_count % 30 == 0:
                print(f"Downloaded {downloaded_count} images...")

        except Exception:
            continue

    print(f"\n✅ {artist_name.upper()} scraping complete. Total downloaded images: {downloaded_count}")


# --- EXECUTION ---
if __name__ == "__main__":
    for source in ARTIST_SOURCES:
        scrape_images(source)
    print("\n\n--- ALL DOWNLOADS COMPLETE ---")
    print("NEXT STEP: Run your 01_preprocess_data.py script to standardize the files.")