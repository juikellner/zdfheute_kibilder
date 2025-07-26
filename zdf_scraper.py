import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import openai
import replicate
import re

# Load API keys from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN")

# Streamlit app title
st.set_page_config(layout="wide")
st.title("📰 ZDFheute KI-Teaser")

# Hinweistext (klein und responsiv)
st.markdown("<p style='font-size: 1.1rem; line-height: 1.4;'>🔍 Diese Anwendung scrapt die drei Top-Teaser auf zdfheute.de und nutzt GPT-4o/4 zur Bildbeschreibung und Prompt-Erstellung basierend auf dem Bildinhalt, der Schlagzeile, der Dachzeile und analysierten Informationen aus der Bild-URL eines Artikels. Für die Bildgenerierung wird das Modell <code>google/imagen-4-fast</code> sowie <code>luma/photon-flash</code> auf replicate.com verwendet.</p>", unsafe_allow_html=True)

# GPT-gestützte Extraktion von Kontext aus Bild-URL
def extract_context_from_url(url):
    try:
        filename = url.split("/")[-1].split("~")[0]
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein Nachrichtensystem. Deine Aufgabe ist es, aus einem Bild-Dateinamen (wie in einer URL) sinnvolle kontextuelle Informationen wie Personennamen, Orte, Länder oder Ereignisse zu extrahieren und in einen sinnvollen Nachrichtenzusammenhang zu setzen. Antworte mit einem vollständigen, aber kurzen Satz im journalistischen Stil."},
                {"role": "user", "content": f"Dateiname aus URL: {filename}"}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"GPT-Kontextanalyse fehlgeschlagen: {e}")
        return ""

# Scrape top news articles from ZDFheute
def scrape_top_articles():
    url = "https://www.zdfheute.de/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        teasers = soup.find_all("picture", class_="slrzex8")[:3]
        results = []
        for pic in teasers:
            img = pic.find("img")
            if not img:
                continue
            srcset = img.get("srcset", "")
            images = [s.strip().split(" ")[0] for s in srcset.split(",") if "https://" in s]
            filtered = [s for s in images if re.search(r"~(\d+)x(\d+)", s)]
            img_url = filtered[-1] if filtered else images[-1]
            parent = pic.find_parent("div")
            while parent and not parent.find("a"):
                parent = parent.find_parent("div")
            if parent:
                a_tag = parent.find("a")
                title = a_tag.get_text(strip=True)
                article_url = "https://www.zdfheute.de" + a_tag["href"]
                dachzeile_tag = parent.find("span")
                dachzeile = dachzeile_tag.get_text(strip=True) if dachzeile_tag else ""
            else:
                title, dachzeile, article_url = "", "", ""
            results.append({"image_url": img_url, "headline": title, "dachzeile": dachzeile, "url": article_url})
        return results
    except Exception as e:
        st.error(f"Fehler beim Scraping: {e}")
        return []

# Prompt + Beschreibung generieren
def generate_prompt(headline, dachzeile, image_url):
    try:
        context = extract_context_from_url(image_url)
        vision_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Du bist ein visuelles Analysemodell. Du beschreibst journalistische Nachrichtenbilder in Stichpunkten. Berücksichtige unbedingt den folgenden Kontext aus der Bild-URL: '{context}'. Binde diesen Kontext in die Beschreibung ein."},
                {"role": "user", "content": "Analysiere das folgende Bild."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ],
            max_tokens=1000
        )
        image_description = vision_response.choices[0].message.content.strip().replace("\n", " ")
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein kreativer Prompt-Designer für Text-zu-Bild-KI im News-Bereich."},
                {"role": "user", "content": f"Erstelle einen photo-realistischen Bildprompt auf Englisch für folgende ZDF-Schlagzeile: '{headline}'\nDachzeile: '{dachzeile}'\nKontext: '{context}'\nBildbeschreibung: {image_description}. Der Prompt soll für ein Bildmodell geeignet sein und darf keinen Text enthalten."}
            ]
        )
        return response.choices[0].message.content.strip(), image_description
    except Exception as e:
        st.error(f"Fehler bei Prompt-Erstellung: {e}")
        return None, None

# Generiere beide Bilder mit Replicate
def generate_image_url(prompt):
    try:
        os.environ["REPLICATE_API_TOKEN"] = replicate_token
        imagen_output = replicate.run("google/imagen-4-fast", {"prompt": prompt, "aspect_ratio": "4:3", "output_format": "jpg", "safety_filter_level": "block_only_high"})
        imagen_result = imagen_output[0] if isinstance(imagen_output, list) else imagen_output
        luma_output = replicate.run("luma/photon-flash", {"prompt": prompt, "aspect_ratio": "16:9", "image_reference_weight": 0.85, "style_reference_weight": 0.85})
        luma_result = luma_output.url() if hasattr(luma_output, 'url') else luma_output[0] if isinstance(luma_output, list) else luma_output
        return str(imagen_result), str(luma_result)
    except Exception as e:
        st.error(f"Fehler bei Bildgenerierung: {e}")
        return None, None

# App Start
data = scrape_top_articles()
if data:
    for idx, item in enumerate(data):
        st.markdown("---")
        st.markdown(f"### {item['headline']}")
        st.markdown(f"**{item['dachzeile']}**")
        st.markdown(f"🔗 [Zum Artikel]({item['url']})")
        if f"generated_{idx}" not in st.session_state:
            st.session_state[f"generated_{idx}"] = {"prompt": None, "imagen_url": None, "luma_url": None, "image_description": None}
        st.markdown("**🌐 Bildquelle (URL):**")
        st.markdown(f"<code style='font-size: 0.9rem'>{item['image_url']}</code>", unsafe_allow_html=True)
        if not st.session_state[f"generated_{idx}"]["image_description"]:
            with st.spinner("📷 Analysiere Bild..."):
                _, description = generate_prompt(item['headline'], item['dachzeile'], item['image_url'])
                st.session_state[f"generated_{idx}"]["image_description"] = description
        if st.session_state[f"generated_{idx}"].get("image_description"):
            st.markdown("**🖼️ Bildbeschreibung:**")
            st.markdown(f"<code style='font-size: 1.0rem'>{st.session_state[f'generated_{idx}']['image_description']}</code>", unsafe_allow_html=True)
        if st.button(f"✨ Prompt & Bild generieren für: {item['headline']}", key=f"btn_generate_{idx}"):
            with st.spinner("🔍 Erzeuge Prompt..."):
                prompt, _ = generate_prompt(item['headline'], item['dachzeile'], item['image_url'])
                st.session_state[f"generated_{idx}"]["prompt"] = prompt
            if prompt:
                st.markdown("**📝 Generierter Prompt:**")
                st.markdown(f"<code style='font-size: 1.0rem'>{prompt}</code>", unsafe_allow_html=True)
                with st.spinner("🎨 Erzeuge KI-Bilder..."):
                    imagen_url, luma_url = generate_image_url(prompt)
                    st.session_state[f"generated_{idx}"]["imagen_url"] = imagen_url
                    st.session_state[f"generated_{idx}"]["luma_url"] = luma_url
            st.rerun()
        generated = st.session_state.get(f"generated_{idx}", {})
        prompt = generated.get("prompt")
        imagen_url = generated.get("imagen_url")
        luma_url = generated.get("luma_url")
        if imagen_url or luma_url:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(item["image_url"], caption="Originalbild", use_column_width=True)
            with col2:
                if imagen_url:
                    try:
                        response = requests.get(imagen_url)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            st.image(img, caption="KI-Bild: google/imagen-4-fast", width=350)
                        else:
                            st.warning("⚠️ Imagen-Bild konnte nicht geladen werden.")
                    except Exception as e:
                        st.warning(f"⚠️ Fehler beim Laden des Imagen-Bildes: {e}")
                if luma_url:
                    try:
                        response = requests.get(luma_url)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            st.image(img, caption="KI-Bild: luma/photon-flash", width=350)
                        else:
                            st.warning("⚠️ Luma-Bild konnte nicht geladen werden.")
                    except Exception as e:
                        st.warning(f"⚠️ Fehler beim Laden des Luma-Bildes: {e}")
                else:
                    st.warning("⚠️ Kein gültiges Bild von luma/photon-flash empfangen.")
else:
    st.warning("Keine Daten gefunden.")
