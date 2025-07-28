# (Teil 1 – Imports & Setup)
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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN")

st.set_page_config(layout="wide")
st.title("📰 ZDFheute KI-Teaser")

st.markdown("<p style='font-size: 1.1rem;'>🔍 Diese Anwendung scrapt die drei Top-Teaser auf zdfheute.de, nutzt GPT-4 zur Bildanalyse und erstellt passende KI-Bilder mit replicate.com.</p>", unsafe_allow_html=True)

# (Teil 2 – Kontextanalyse aus URL)
def extract_context_from_url(url):
    try:
        filename = url.split("/")[-1].split("~")[0]
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extrahiere aus einem Bild-Dateinamen relevante Namen, Orte oder Ereignisse im Nachrichtenkontext."},
                {"role": "user", "content": f"Dateiname aus URL: {filename}"}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"GPT-Kontextanalyse fehlgeschlagen: {e}")
        return ""

# (Teil 3 – Scraping-Logik mit Video-Filter)
def scrape_top_articles():
    url = "https://www.zdfheute.de/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        teasers = soup.find_all("picture", class_="slrzex8")

        results = []
        for pic in teasers:
            # ⛔️ Skip if 'mit Video' Hinweis vorhanden
            if pic.find_next("div", class_="hiqr0m7"):
                continue

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

            results.append({
                "image_url": img_url,
                "headline": title,
                "dachzeile": dachzeile,
                "url": article_url
            })

            if len(results) == 3:
                break

        return results

    except Exception as e:
        st.error(f"Fehler beim Scraping: {e}")
        return []

# (Teil 4 – Promptgenerierung)
def generate_prompt(headline, dachzeile, image_url):
    try:
        context = extract_context_from_url(image_url)
        vision_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Beschreibe das Nachrichtenbild unter Einbezug dieses Kontexts: '{context}'."},
                {"role": "user", "content": "Analysiere das folgende Bild."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ],
            max_tokens=1000
        )
        image_description = vision_response.choices[0].message.content.strip().replace("\n", " ")

        prompt_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein kreativer Prompt-Designer für text-to-image KI."},
                {"role": "user", "content": f"Erstelle einen fotorealistischen Prompt für: '{headline}'\nDachzeile: '{dachzeile}'\nKontext: '{context}'\nBildbeschreibung: {image_description}"}
            ]
        )
        return prompt_response.choices[0].message.content.strip(), image_description
    except Exception as e:
        st.error(f"Fehler bei Prompt-Erstellung: {e}")
        return None, None

# (Teil 5 – Bildgenerierung)
def generate_image_url(prompt):
    try:
        os.environ["REPLICATE_API_TOKEN"] = replicate_token
        imagen_output = replicate.run("google/imagen-4-fast", {
            "prompt": prompt, "aspect_ratio": "4:3",
            "output_format": "jpg", "safety_filter_level": "block_only_high"
        })
        luma_output = replicate.run("luma/photon-flash", {
            "prompt": prompt, "aspect_ratio": "16:9",
            "image_reference_weight": 0.85, "style_reference_weight": 0.85
        })
        imagen_result = imagen_output[0] if isinstance(imagen_output, list) else imagen_output
        luma_result = luma_output.get("image") if isinstance(luma_output, dict) else luma_output[0]
        return str(imagen_result), str(luma_result)
    except Exception as e:
        st.error(f"Fehler bei Bildgenerierung: {e}")
        return None, None

# (Teil 6 – Streamlit App Flow)
data = scrape_top_articles()
if data:
    for idx, item in enumerate(data):
        st.markdown("---")
        st.markdown(f"### {item['headline']}")
        st.markdown(f"**{item['dachzeile']}**")
        st.markdown(f"🔗 [Zum Artikel]({item['url']})")

        if f"generated_{idx}" not in st.session_state:
            st.session_state[f"generated_{idx}"] = {
                "prompt": None, "imagen_url": None,
                "luma_url": None, "image_description": None
            }

        st.markdown("**🌐 Bildquelle (URL):**")
        st.code(item["image_url"])

        if not st.session_state[f"generated_{idx}"]["image_description"]:
            with st.spinner("📷 Analysiere Bild..."):
                _, description = generate_prompt(item['headline'], item['dachzeile'], item['image_url'])
                st.session_state[f"generated_{idx}"]["image_description"] = description

        if st.session_state[f"generated_{idx}"].get("image_description"):
            st.markdown("**🖼️ Bildbeschreibung:**")
            st.code(st.session_state[f"generated_{idx}"]["image_description"])

        if st.button(f"✨ Prompt & Bild generieren für: {item['headline']}", key=f"btn_{idx}"):
            with st.spinner("✏️ Erzeuge Prompt..."):
                prompt, _ = generate_prompt(item['headline'], item['dachzeile'], item['image_url'])
                st.session_state[f"generated_{idx}"]["prompt"] = prompt

            if prompt:
                st.markdown("**📝 Generierter Prompt:**")
                st.code(prompt)
                with st.spinner("🎨 Generiere KI-Bilder..."):
                    imagen_url, luma_url = generate_image_url(prompt)
                    st.session_state[f"generated_{idx}"]["imagen_url"] = imagen_url
                    st.session_state[f"generated_{idx}"]["luma_url"] = luma_url
            st.rerun()

        generated = st.session_state.get(f"generated_{idx}")
        if generated["imagen_url"] or generated["luma_url"]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(item["image_url"], caption="Originalbild", use_column_width=True)
            with col2:
                if generated["imagen_url"]:
                    st.image(generated["imagen_url"], caption="KI-Bild: google/imagen-4-fast", width=350)
                if generated["luma_url"]:
                    st.image(generated["luma_url"], caption="KI-Bild: luma/photon-flash", width=350)
else:
    st.warning("⚠️ Keine Artikel gefunden.")
