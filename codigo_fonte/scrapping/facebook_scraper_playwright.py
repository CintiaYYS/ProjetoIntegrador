from playwright.sync_api import sync_playwright
import requests
import os
import pandas as pd
import re
from urllib.parse import urlparse
from pathlib import Path
from PIL import Image
from io import BytesIO



def scrape_facebook_post(post_url):
    save_dir = "dados/brutos/imagens"
    largura_min = 300
    altura_min = 300

    os.makedirs(save_dir, exist_ok=True)

    # Temporariamente inicia o navegador apenas se necessário
    image_url = None
    file_name = None
    saved_image_path = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Adiciona cabeçalho para evitar bloqueio do Facebook
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        })

        try:
            # Espera 20 segundos para carregamento da página
            page.goto(post_url, timeout=20000)
            # O código espera 5 segundos
            page.wait_for_timeout(5000)

            # Coleta todas as imagens visíveis
            image_urls = page.eval_on_selector_all(
                "img", 
                "elements => elements.map(e => ({src: e.src, width: e.naturalWidth, height: e.naturalHeight}))"
            )

            # Seleciona a maior imagem válida
            max_area = 0
            for img in image_urls:
                if img['width'] >= largura_min and img['height'] >= altura_min:
                    area = img['width'] * img['height']
                    if area > max_area:
                        max_area = area
                        image_url = img['src']
        except Exception as e:
            print(f"[!] Erro ao acessar {post_url}: {e}")
            browser.close()
            return None
        
        browser.close()

    # Se encontramos uma imagem válida
    if image_url:
        file_name = os.path.basename(urlparse(image_url).path)
        saved_image_path = os.path.join(save_dir, file_name)

        # Se a imagem já estiver na pasta, retorna diretamente
        if os.path.exists(saved_image_path):
            print(f"[✓] Imagem já existe: {file_name}")
            return file_name

        # Caso contrário, baixa
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img.save(saved_image_path)
            return file_name
        except Exception as e:
            print(f"[!] Erro ao salvar imagem: {e}")
            return None

    return None



def baixarImagens():

    # Caminho do CSV
    csv_path = Path("dados/processados/base_posts.csv")
    
    
    # Lê o arquivo CSV
    df = pd.read_csv(csv_path)
    
    
    nomes_imagens = []

    for i, row in df.iterrows():
        url = row['Url']
        print(f"[{i+1}/{len(df)}] Processando URL: {url}")
        nome_imagem = scrape_facebook_post(url)
        nomes_imagens.append(nome_imagem)

    # Adiciona a coluna com os nomes das imagens
    df['nome_imagem'] = nomes_imagens

    # Remove as linhas onde a imagem não foi baixada
    df = df.dropna(subset=['nome_imagem']).reset_index(drop=True)
    
    # Salva o DataFrame atualizado
    df.to_csv(csv_path, index=False)
    