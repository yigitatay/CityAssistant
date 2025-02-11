from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

chrome_path = "/usr/local/bin/chrome-for-testing/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
options = Options()
options.binary_location = chrome_path

service = Service('./chromedriver') 
driver = webdriver.Chrome(service=service, options=options)

url = 'https://311.dc.gov/citizen/s/'
driver.get(url)

try:
    wait = WebDriverWait(driver, 10)
    clickable_element = wait.until(
EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-sr-name="24-allCurrent"]'))    )
    
    clickable_element.click()
    
    time.sleep(3)
    
    updated_html = driver.page_source
    
    soup = BeautifulSoup(updated_html, 'html.parser')
    ul_element = soup.find('ul', class_='suggest')
    
    # Extract and print the text from each span inside li
    for li in ul_element.find_all('li'):
        span = li.find('span', class_='childserviceTypeName')
        if span:
            print(span.get_text(strip=True))
finally:
    driver.quit()
