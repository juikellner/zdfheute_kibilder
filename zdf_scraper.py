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
st.markdown("<p style='font-size: 1.1rem; line-height: 1.4;'>🔍 Diese Anwendung scrapt die drei Top-Teaser auf zdfheute.de und nutzt GPT-4o/4 zur Bildbeschreibung und Prompt-Erstellung basierend auf dem Bildinhalt, der Schlagzeile, der Dachzeile und analysierten Informationen aus der Bild-URL eines Artikels. Für die Bildgenerierung wird das Modell <code>google/imagen-4-fast</code> auf replicate.com verwendet.</p>", unsafe_allow_html=True)

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

# Scrape top news articles from ZDFheute with best image resolution

def scrape_top_articles():
    url = "https://www.zdfheute.de/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        h2_blocks = soup.find_all("h2", class_="hvzzigm")
        results = []

        for h2 in h2_blocks[:3]:
            spans = h2.find_all("span")
            if len(spans) >= 3:
                dachzeile = spans[0].get_text(strip=True)
                headline = spans[2].get_text(strip=True)
                a_tag = spans[2].find("a")
                url_suffix = a_tag["href"] if a_tag else ""
                full_url = "https://www.zdfheute.de" + url_suffix

                picture = h2.find_previous("picture")
                img_url = ""
                if picture:
                    img = picture.find("img")
                    if img and img.get("srcset"):
                        images = [s.strip().split(" ")[0] for s in img["srcset"].split(",") if "https://" in s]
                        filtered = [s for s in images if re.search(r"~(\\d+)x(\\d+)", s)]
                        img_url = filtered[-1] if filtered else images[-1]

                results.append({
                    "headline": headline,
                    "dachzeile": dachzeile,
                    "url": full_url,
                    "image_url": img_url
                })

        return results

    except Exception as e:
        st.error(f"Fehler beim Scraping: {e}")
        return []

# Generate image prompt using OpenAI

def generate_prompt(headline, dachzeile, image_url):
    try:
        context_from_url = extract_context_from_url(image_url)

        vision_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Du bist ein visuelles Analysemodell. Du beschreibst journalistische Nachrichtenbilder in Stichpunkten. Berücksichtige unbedingt den folgenden Kontext aus der Bild-URL: '{context_from_url}'. Binde diesen Kontext in die Beschreibung ein."},
                {"role": "user", "content": "Analysiere das folgende Bild und beschreibe den visuellen Inhalt unter Einbeziehung des Kontexts."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ],
            max_tokens=1000
        )
        image_description = vision_response.choices[0].message.content.strip().replace("\n", " ")

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein kreativer Prompt-Designer für Text-zu-Bild-KI im News-Bereich."},
                {"role": "user", "content": f"Erstelle einen photo-realistischen Bildprompt auf Englisch für folgende ZDF-Schlagzeile: '{headline}'\nDachzeile: '{dachzeile}'\nKontext: '{context_from_url}'\nBildbeschreibung: {image_description}. Der Prompt soll für ein Bildmodell geeignet sein und darf keinen Text enthalten."}
            ]
        )
        return response.choices[0].message.content.strip().replace("\n", " "), image_description
    except Exception as e:
        st.error(f"Fehler bei Prompt-Erstellung: {e}")
        return None, None

# Generate image with Replicate

def generate_image_url(prompt):
    try:
        os.environ["REPLICATE_API_TOKEN"] = replicate_token
        output = replicate.run(
            "google/imagen-4-fast",
            input={
                "prompt": prompt,
                "aspect_ratio": "4:3",
                "output_format": "jpg",
                "safety_filter_level": "block_only_high"
            }
        )
        result = output[0] if isinstance(output, list) and len(output) > 0 else output
        return str(result)
    except Exception as e:
        st.error(f"Fehler bei Bildgenerierung: {e}")
        return None

# MAIN APP FLOW

data = scrape_top_articles()

if data:
    for idx, item in enumerate(data):
        st.markdown("---")
        st.markdown(f"### {item['headline']}")
        st.markdown(f"**{item['dachzeile']}**")
        st.markdown(f"🔗 [Zum Artikel]({item['url']})")

        if f"generated_{idx}" not in st.session_state:
            st.session_state[f"generated_{idx}"] = {"prompt": None, "image_url": None, "image_description": None}

        st.image(item["image_url"], caption="Originalbild", width=800)

        st.markdown("**🌐 Bildquelle (URL):**")
        st.markdown(f"<code style='font-size: 0.9rem; word-break: break-word; white-space: pre-wrap;'>{item['image_url']}</code>", unsafe_allow_html=True)

        if not st.session_state[f"generated_{idx}"]["image_description"]:
            with st.spinner("📷 Analysiere Bild..."):
                _, image_description = generate_prompt(item['headline'], item['dachzeile'], item['image_url'])
                st.session_state[f"generated_{idx}"]["image_description"] = image_description

        if st.session_state[f"generated_{idx}"].get("image_description"):
            st.markdown("**🖼️ Bildbeschreibung:**")
            st.markdown(f"<code style='font-size: 1.0rem; word-break: break-word; white-space: pre-wrap;'>{st.session_state[f'generated_{idx}']['image_description']}</code>", unsafe_allow_html=True)

        if st.button(f"✨ Prompt & Bild generieren für: {item['headline']}", key=f"btn_generate_{idx}"):
            with st.spinner("🔍 Erzeuge Prompt..."):
                prompt, _ = generate_prompt(item['headline'], item['dachzeile'], item['image_url'])
                st.session_state[f"generated_{idx}"]["prompt"] = prompt

            if prompt:
                st.markdown("**📝 Generierter Prompt:**")
                st.markdown(f"<code style='font-size: 1.0rem; word-break: break-word; white-space: pre-wrap;'>{prompt}</code>", unsafe_allow_html=True)

                with st.spinner("🎨 Erzeuge KI-Bild..."):
                    image_url = generate_image_url(prompt)
                    st.session_state[f"generated_{idx}"]["image_url"] = image_url
            st.rerun()

        generated = st.session_state.get(f"generated_{idx}", {})
        prompt = generated.get("prompt")
        image_url = generated.get("image_url")

        if prompt:
            st.markdown("**📝 Generierter Prompt:**")
            st.markdown(f"<code style='font-size: 1.0rem; word-break: break-word; white-space: pre-wrap;'>{prompt}</code>", unsafe_allow_html=True)

        if image_url:
            col1, col2 = st.columns(2)
            with col1:
                st.image(item["image_url"], caption="Originalbild", width=800)
            with col2:
                st.image(image_url, caption="KI-generiertes Bild", width=800)
        elif prompt:
            st.info("⚠️ Kein KI-Bildlink von Replicate erhalten.")
else:
    st.warning("Keine Daten gefunden.")