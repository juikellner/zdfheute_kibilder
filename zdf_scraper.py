import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
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
st.markdown("<p style='font-size: 0.8rem; line-height: 1.4;'>🔍 Diese Anwendung nutzt GPT-4o zur Prompt-Erstellung basierend auf dem Bildinhalt, der Schlagzeile, der Dachzeile, der Bildbeschreibung und analysierten Informationen aus der Bild-URL eines Artikels. Für die Bildgenerierung wird das Modell <code>google/imagen-4-fast</code> von Replicate verwendet. Das erzeugte Bild enthält keinen Text.</p>", unsafe_allow_html=True)

# Kontext aus Bild-URL extrahieren
def extract_context_from_url(url):
    filename = url.split("/")[-1].split("~")[0]
    parts = re.split("[-_]+", filename)
    keywords = [part for part in parts if part.isalpha() and len(part) > 2]
    return ", ".join(keywords)

# Scrape top news articles from ZDFheute with best image resolution (min. 276x155)
def scrape_top_articles():
    url = "https://www.zdfheute.de/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
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
            images = [s.strip().split(" ")[0] for s in srcset.split(",") if s.strip() and "https://" in s]

            filtered_images = []
            for img_url in images:
                dims_match = re.search(r"~(\d+)x(\d+)", img_url)
                if dims_match:
                    try:
                        width, height = map(int, dims_match.groups())
                        if width >= 276 and height >= 155:
                            filtered_images.append((width, img_url))
                    except ValueError:
                        continue

            if not filtered_images:
                continue

            img_url = sorted(filtered_images, key=lambda x: x[0], reverse=True)[0][1]

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
                title = "Kein Titel gefunden"
                dachzeile = ""
                article_url = ""

            context = extract_context_from_url(img_url)

            results.append({
                "image_url": img_url,
                "headline": title,
                "dachzeile": dachzeile,
                "url": article_url,
                "context": context
            })
        return results
    except Exception as e:
        st.error(f"Fehler beim Scraping: {e}")
        return []

# --- Hauptanwendung ---
data = scrape_top_articles()
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = {}

if data:
    for idx, item in enumerate(data):
        st.markdown("---")
        st.markdown(f"### {item['headline']}")
        st.markdown(f"**{item['dachzeile']}**")
        st.markdown(f"🔗 [Zum Artikel]({item['url']})")

        st.image(item["image_url"], caption="Originalbild (ZDFheute)", width=800)

        with st.spinner("Analysiere Originalbild..."):
            try:
                vision_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Du bist ein kreativer Prompt-Designer. Beschreibe das Bild stichpunktartig."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": item["image_url"]}},
                                {"type": "text", "text": "Bitte analysiere dieses Bild."}
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                image_description = vision_response.choices[0].message.content.strip()
            except Exception as e:
                st.warning(f"Bildbeschreibung konnte nicht erstellt werden: {e}")
                image_description = ""

        if image_description:
            st.markdown("**📝 Bildbeschreibung:**")
            st.markdown(f"<code style='font-size: 1rem; white-space: normal;'>{image_description}</code>", unsafe_allow_html=True)

        if st.button(f"✨ Prompt & Bild generieren ...", key=f"btn_generate_{idx}"):
            with st.spinner("Erstelle Prompt..."):
                try:
                    prompt_response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "Du bist ein kreativer Prompt-Designer für Text-zu-Bild-KI."},
                            {"role": "user", "content": f"Erstelle einen filmisch-realistischen Prompt auf Englisch für folgende ZDF-Schlagzeile: '{item['headline']}'\nDachzeile: '{item['dachzeile']}'\nKontext: '{item['context']}'\nBildbeschreibung: '{image_description}'. Der Prompt soll für das Modell 'google/imagen-4-fast' geeignet sein."}
                        ]
                    )
                    prompt = prompt_response.choices[0].message.content.strip()
                    st.markdown("**🎯 Generierter Prompt:**")
                    st.markdown(f"<code style='font-size: 1rem; white-space: normal;'>{prompt}</code>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Fehler bei Prompt-Erstellung: {e}")
                    prompt = None

            if prompt:
                with st.spinner("Erzeuge KI-Bild..."):
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
                        img_url = str(output[0]) if isinstance(output, list) else str(output)
                        st.image(img_url, caption="KI-generiertes Bild (Replicate)", width=800)
                    except Exception as e:
                        st.error(f"Fehler bei der Bildgenerierung: {e}")
else:
    st.warning("Keine Daten gefunden.")
