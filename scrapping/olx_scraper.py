from playwright.sync_api import sync_playwright
import time
import pandas as pd
import random
import logging
import unicodedata

logger = logging.getLogger(__name__)

URL_TARGET = "https://www.olx.com.br/autos-e-pecas/carros-vans-e-utilitarios"
NUM_PAGES = 100


def safe_query(element, selector, method="inner_text"):
    """Safely query an element and return its text or attribute."""
    try:
        res = element.query_selector(selector)
        if res is None:
            return None
        if method == "inner_text":
            return res.inner_text().strip()
        elif method == "get_attribute":
            return res.get_attribute("href")
    except Exception:
        return None

def extract_car_listing_data(car_element):
    """Extract data from a single car listing element."""
    url = safe_query(car_element, "a", method="get_attribute")
    title = safe_query(car_element, "h2[class^='typo-body-large']")
    km = safe_query(car_element, "[aria-label$='quilômetros rodados']")
    color = safe_query(car_element, "[aria-label^='Cor']")
    motor = safe_query(car_element, "[aria-label^='Motor']")
    price = safe_query(car_element, "h3[class^='typo-body-large']")
    
    return {
        "url": url,
        "title_list": title,
        "km_list": km,
        "color_list": color,
        "motor_list": motor,
        "price_list": price
    }


def scrape_listings_from_page(page, page_number):
    """Scrape all car listings from a single page."""
    url = f"{URL_TARGET}?o={page_number}"
    logger.info(f"[Página {page_number}/{NUM_PAGES}] Iniciando scraping da página")

    try:
        page.goto(url, wait_until="domcontentloaded")
        logger.debug(f"[Página {page_number}/{NUM_PAGES}] Página carregada com sucesso")
    except Exception as e:
        logger.error(f"[Página {page_number}/{NUM_PAGES}] Erro ao carregar página: {e}")
        return []

    car_elements = page.query_selector_all("div[class^='olx-adcard__content']")
    logger.info(f"[Página {page_number}/{NUM_PAGES}] Encontrados {len(car_elements)} anúncios")

    car_data = []
    for idx, car_element in enumerate(car_elements, 1):
        listing_data = extract_car_listing_data(car_element)
        car_data.append(listing_data)
        title = listing_data.get('title_list', 'Sem título')
        logger.debug(f"[Página {page_number}] Anúncio {idx}/{len(car_elements)}: {title}")
    
    sleep_time = random.uniform(2, 5)
    logger.debug(f"[Página {page_number}] Aguardando {sleep_time:.2f}s antes da próxima página")
    time.sleep(sleep_time)
    return car_data


def scrape_olx_list(page):
    """Scrape car listings from multiple pages."""
    logger.info(f"[Listagens] Iniciando scraping de {NUM_PAGES} página(s)")
    car_data = []

    for page_number in range(1, NUM_PAGES + 1):
        page_listings = scrape_listings_from_page(page, page_number)
        car_data.extend(page_listings)
        logger.info(f"[Listagens] Página {page_number} processada: {len(page_listings)} anúncios coletados")

    logger.info(f"[Listagens] Concluído. Total de anúncios encontrados: {len(car_data)}")
    return car_data

def extract_description(page):
    """Extract description from car details page."""
    try:
        descricao_element = page.locator('[data-section="description"]').inner_text(timeout=5000)
        logger.debug("[Detalhes] Descrição extraída com sucesso")
        return descricao_element
    except Exception as e:
        logger.warning(f"[Detalhes] Falha ao extrair descrição: {e}")
        return None


def extract_car_details(page):
    """Extract technical details from car details page."""
    details = {}
    try:
        car_details_element = page.locator('#details [data-ds-component="DS-Container"]').all()
        logger.debug(f"[Detalhes] Encontrados {len(car_details_element)} elementos de detalhes")
        
        for detail in car_details_element:
            try:
                key_span = detail.locator('span[data-variant="overline"]').first
                key = key_span.inner_text(timeout=2000)

                valor_container = detail.locator('div[class^="ad__sc-2h9gkk-1"]').first
                valor_span_a = valor_container.locator('span:not([data-variant="overline"]), a').last
                valor = valor_span_a.inner_text(timeout=2000)

                if key and valor:
                    key = key.lower().replace(" ", "_")
                    details[key] = valor
                    logger.debug(f"[Detalhes] Extraído: {key} = {valor}")
            except Exception as e:
                logger.debug(f"[Detalhes] Falha ao extrair detalhe individual: {e}")
                continue
    except Exception as e:
        logger.warning(f"[Detalhes] Falha ao extrair detalhes do carro: {e}")
    
    return details


def normalize_option_name(option_text):
    """Normalize option name by removing accents and special characters."""
    # Normalize the option name: remove accents and convert to lowercase
    normalized_key = unicodedata.normalize('NFD', option_text.lower())
    normalized_key = ''.join(c for c in normalized_key if unicodedata.category(c) != 'Mn')
    # Replace spaces with underscores
    normalized_key = normalized_key.replace(" ", "_")
    # Remove special characters, keep only alphanumeric and underscores
    normalized_key = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized_key)
    # Remove multiple consecutive underscores
    while '__' in normalized_key:
        normalized_key = normalized_key.replace('__', '_')
    return normalized_key.strip('_')


def extract_car_options(page):
    """Extract car options/features from car details page."""
    options = {}
    car_options_element = page.locator('div[class^="ad__sc-1jr3zuf-1"]').all()
    logger.debug(f"[Opcionais] Encontrados {len(car_options_element)} elementos de opcionais")
    
    for option in car_options_element:
        try:
            key_span = option.inner_text(timeout=2000)
            if key_span:
                # Split by newlines in case multiple options are concatenated
                option_texts = [opt.strip() for opt in key_span.split('\n') if opt.strip()]
                
                for opt_text in option_texts:
                    normalized_key = normalize_option_name(opt_text)
                    if normalized_key:
                        options[normalized_key] = True
                        logger.debug(f"[Opcionais] Opcional encontrado: {opt_text} -> {normalized_key}")
        except Exception as e:
            logger.debug(f"[Opcionais] Falha ao extrair opcional: {e}")
            continue
    
    logger.debug(f"[Opcionais] Total de opcionais extraídos: {len(options)}")
    return options


def parse_location_string(location_string):
    """Parse location string into city, state, and zip code."""
    city = state = zip_code = None
    
    if not location_string:
        return city, state, zip_code
    
    city_state_zip_split = [item.strip() for item in location_string.split(",")]
    
    if len(city_state_zip_split) == 3:
        city, state, zip_code = city_state_zip_split
    elif len(city_state_zip_split) == 2:
        city, state = city_state_zip_split
        zip_code = None
    elif len(city_state_zip_split) == 1:
        city = city_state_zip_split[0]
        state = zip_code = None
    
    return city, state, zip_code


def extract_location(page):
    """Extract location information from car details page."""
    location_data = {
        'neighborhood': None,
        'city': None,
        'state': None,
        'zip_code': None
    }
    
    try:
        location_element = page.locator('div[class$="gYzJpw"]')

        # Extract neighborhood
        try:
            neighborhood = location_element.locator("span.olx-text--body-medium").inner_text(timeout=2000)
            location_data['neighborhood'] = neighborhood
            logger.debug(f"[Localização] Bairro extraído: {neighborhood}")
        except Exception:
            logger.debug("[Localização] Bairro não encontrado")

        # Extract city, state, zip code
        try:
            city_state_zip_raw = location_element.locator("span.olx-text--body-small").inner_text(timeout=2000)
            city, state, zip_code = parse_location_string(city_state_zip_raw)
            location_data['city'] = city
            location_data['state'] = state
            location_data['zip_code'] = zip_code
            logger.debug(f"[Localização] Localização extraída: {city}, {state}, {zip_code}")
        except Exception as e:
            logger.debug(f"[Localização] Falha ao extrair string de localização: {e}")
    except Exception as e:
        logger.warning(f"[Localização] Falha ao extrair localização: {e}")
    
    return location_data


def scrape_car_details(page, url):
    """Scrape all details from a car details page."""
    logger.debug(f"[Detalhes] Iniciando scraping de detalhes: {url}")
    
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        logger.debug("[Detalhes] Página de detalhes carregada com sucesso")
    except Exception as e:
        logger.warning(f"[Detalhes] Falha ao navegar para página de detalhes {url}: {e}")
        return {}

    all_data = {}
    
    # Extract description
    all_data["description"] = extract_description(page)
    
    # Extract technical details
    details = extract_car_details(page)
    all_data.update(details)
    
    # Extract car options
    options = extract_car_options(page)
    all_data.update(options)
    
    # Extract location
    location = extract_location(page)
    all_data.update(location)
    
    logger.debug(f"[Detalhes] Scraping concluído: {len(all_data)} campos extraídos")
    return all_data

def create_browser_context(playwright):
    """Create and configure browser context."""
    logger.info("[Browser] Inicializando navegador Chromium")
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport={"width": 1920, "height": 1080}
    )
    page = context.new_page()
    logger.info("[Browser] Navegador inicializado com sucesso")
    return browser, page


def scrape_cars_details_batch(page, car_data):
    """Scrape details for a batch of cars."""
    total_cars = len(car_data)
    logger.info(f"[Detalhes] Iniciando scraping de detalhes para {total_cars} carro(s)")
    
    for idx, car in enumerate(car_data, 1):
        if car["url"]:
            title = car.get('title_list', 'Sem título')
            logger.info(f"[Detalhes] Processando carro {idx}/{total_cars}: {title}")
            car_details = scrape_car_details(page, car["url"])
            car.update(car_details)
            sleep_time = random.uniform(2, 3)
            logger.debug(f"[Detalhes] Aguardando {sleep_time:.2f}s antes do próximo carro")
            time.sleep(sleep_time)
        else:
            logger.warning(f"[Detalhes] Carro {idx}/{total_cars} não possui URL, pulando scraping de detalhes")
    
    logger.info(f"[Detalhes] Scraping de detalhes concluído para {total_cars} carro(s)")
    return car_data


def scrape_olx():
    """Main function to scrape OLX car listings and details."""
    logger.info("=" * 60)
    logger.info("[OLX Scraper] Iniciando processo de scraping")
    logger.info("=" * 60)
    
    with sync_playwright() as p:
        browser, page = create_browser_context(p)
        
        try:
            # Scrape listings
            car_data = scrape_olx_list(page)
            
            # Scrape details for each car
            car_data = scrape_cars_details_batch(page, car_data)
        finally:
            logger.info("[Browser] Fechando navegador")
            browser.close()

    logger.info("=" * 60)
    logger.info(f"[OLX Scraper] Processo concluído. Total de carros coletados: {len(car_data)}")
    logger.info("=" * 60)
    return car_data

def save_data(data):
    """Save scraped data to CSV file."""
    total_records = len(data)
    logger.info(f"[Salvamento] Salvando {total_records} registro(s) em CSV")
    
    try:
        # First, collect all option keys (keys with True values)
        all_option_keys = set()
        for record in data:
            for key, value in record.items():
                if value is True:
                    all_option_keys.add(key)
        
        # Ensure all records have all option columns
        normalized_data = []
        for record in data:
            normalized_record = record.copy()
            # Set all option keys to False if not present
            for option_key in all_option_keys:
                if option_key not in normalized_record:
                    normalized_record[option_key] = False
            normalized_data.append(normalized_record)
        
        df = pd.DataFrame(normalized_data)
        file_path = "data/raw/olx_cars.csv"
        df.to_csv(file_path, index=False)
        logger.info(f"[Salvamento] Dados salvos com sucesso em: {file_path}")
        logger.info(f"[Salvamento] Total de colunas: {len(df.columns)}")
        logger.info(f"[Salvamento] Total de colunas de opcionais: {len(all_option_keys)}")
    except Exception as e:
        logger.error(f"[Salvamento] Erro ao salvar dados: {e}")
        raise
