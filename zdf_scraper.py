#%%
import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from openai import OpenAI
import replicate

# Umgebungsvariablen laden
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN") or st.secrets.get("REPLICATE_API_TOKEN")

# Clients initialisieren
openai_client = OpenAI(api_key=openai_api_key)
replicate_client = replicate.Client(api_token=replicate_token)

# ZDFheute-Startseite auslesen
url = "https://www.zdf.de/nachrichten"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

# Container-Elemente finden
articles = soup.find_all("article", class_="news-teaser-item")
data = []

for article in articles[:3]:  # Nur die ersten drei Einträge
    img_tag = article.find("img")
    headline_tag = article.find("span", class_="news-teaser-item__title")
    dachzeile_tag = article.find("span", class_="news-teaser-item__eyecatcher")

    image_url = img_tag["src"] if img_tag and "src" in img_tag.attrs else None
    headline = headline_tag.text.strip() if headline_tag else "[Keine Schlagzeile]"
    dachzeile = dachzeile_tag.text.strip() if dachzeile_tag else "[Keine Dachzeile]"

    if image_url:
        data.append({
            "image_url": image_url,
            "headline": headline,
            "dachzeile": dachzeile
        })

st.set_page_config(page_title="ZDFheute KI-Bilder", layout="wide")
st.title("📰 ZDFheute KI-Bilder Generator")

for idx, entry in enumerate(data):
    st.divider()
    cols = st.columns([1, 2])

    with cols[0]:
        try:
            response = requests.get(entry["image_url"], headers=headers, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Originalbild", use_container_width=True)
            else:
                st.warning(f"Bild konnte nicht geladen werden (Statuscode: {response.status_code})")
        except Exception as e:
            st.error(f"Fehler beim Laden des Bildes: {e}")

    with cols[1]:
        st.subheader(entry.get("headline", "[Keine Schlagzeile]"))
        st.caption(entry.get("dachzeile", "[Keine Dachzeile]"))

        prompt_key = f"prompt_{idx}"
        image_key = f"image_{idx}"

        # Prompt-Formular
        with st.form(key=f"form_prompt_{idx}"):
            submitted_prompt = st.form_submit_button("🎨 Prompt generieren")
            if submitted_prompt:
                with st.spinner("Erstelle Prompt..."):
                    try:
                        completion = openai_client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "Du bist ein Prompt-Experte für fotorealistische, kinoreife Bilder."},
                                {"role": "user", "content": f"Erstelle einen hochwertigen Prompt im Stil von Replicate, basierend auf Dachzeile '{entry['dachzeile']}' und Schlagzeile '{entry['headline']}'."}
                            ]
                        )
                        st.session_state[prompt_key] = completion.choices[0].message.content
                    except Exception as e:
                        st.session_state[prompt_key] = f"Fehler bei Prompt-Erstellung: {e}"

        if prompt_key in st.session_state:
            st.text_area("Prompt", st.session_state[prompt_key], height=200)

        # Bild-Formular
        with st.form(key=f"form_image_{idx}"):
            submitted_image = st.form_submit_button("🧠 Bild generieren")
            if submitted_image and prompt_key in st.session_state:
                with st.spinner("Generiere Bild über Replicate..."):
                    try:
                        output_url = replicate_client.run(
                            "google/imagen-4",
                            input={
                                "prompt": st.session_state[prompt_key],
                                "aspect_ratio": "16:9",
                                "safety_filter_level": "block_medium_and_above"
                            }
                        )
                        st.session_state[image_key] = output_url
                    except Exception as e:
                        st.session_state[image_key] = f"Fehler: {e}"

        if image_key in st.session_state:
            if isinstance(st.session_state[image_key], str) and st.session_state[image_key].startswith("http"):
                st.image(st.session_state[image_key], caption="KI-generiertes Bild", use_container_width=True)
            else:
                st.error(st.session_state[image_key])

# %%
