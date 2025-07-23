#%%
import os
import requests
import replicate
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

# Umgebungsvariablen laden (für lokalen Betrieb)
load_dotenv()

# API-Keys laden (lokal oder über Streamlit Cloud Secrets)
try:
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
    replicate_token = os.getenv("REPLICATE_API_TOKEN") or st.secrets["REPLICATE_API_TOKEN"]
except Exception as e:
    st.error("❌ API-Schlüssel nicht gefunden. Bitte setze sie in deiner .env-Datei oder unter 'Secrets' in Streamlit Cloud.")
    st.stop()

openai_client = OpenAI(api_key=openai_api_key)
replicate_client = replicate.Client(api_token=replicate_token)

st.set_page_config(page_title="ZDF KI-Bilder", layout="wide")
st.title("📰 ZDF Bilder + KI-Visualisierungen")

# Session-State initialisieren
if "entries" not in st.session_state:
    st.session_state.entries = []
if "prompts" not in st.session_state:
    st.session_state.prompts = {}
if "images" not in st.session_state:
    st.session_state.images = {}

def scrape_zdf():
    url = "https://www.zdfheute.de/"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Fehler beim Laden der Webseite: {e}")
        return

    soup = BeautifulSoup(response.content, "html.parser")

    articles = soup.find_all("article", class_="zdf-article-teaser")
    results = []

    for article in articles:
        img_tag = article.find("img")
        dachzeile_tag = article.find("span", class_="o1ximh7k")
        headline_tag = article.find("a", class_="b7kurox")

        if img_tag and dachzeile_tag and headline_tag:
            img_url = img_tag.get("src")
            dachzeile = dachzeile_tag.text.strip()
            headline = headline_tag.text.strip()

            results.append({
                "img_url": img_url,
                "dachzeile": dachzeile,
                "headline": headline
            })

    st.session_state.entries = results

# Scrape Button anzeigen
if st.button("🔄 ZDF-Daten laden"):
    with st.spinner("ZDF-Bilder werden geladen..."):
        scrape_zdf()

if not st.session_state.entries:
    st.info("⬆️ Klicke oben auf 'ZDF-Daten laden', um die Inhalte zu laden.")

for idx, entry in enumerate(st.session_state.entries):
    with st.container():
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(entry["img_url"], caption="Originalbild", use_container_width=True)
        with cols[1]:
            st.subheader(entry["headline"])
            st.caption(entry.get("dachzeile", ""))

            prompt_key = f"prompt_{idx}"
            image_key = f"image_{idx}"

            if st.button(f"🧠 Prompt generieren {idx+1}"):
                with st.spinner("Generiere Bildbeschreibung mit OpenAI..."):
                    prompt_text = f"Erstelle eine cinematische, fotorealistische Bildbeschreibung zur Schlagzeile '{entry['headline']}' und der Dachzeile '{entry['dachzeile']}'."                    
                    try:
                        chat_response = openai_client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "Du erstellst detaillierte Bildbeschreibungen im Stil eines Kamera-Regieanweisungstextes für einen KI-Bildgenerator."},
                                {"role": "user", "content": prompt_text}
                            ]
                        )
                        prompt = chat_response.choices[0].message.content.strip()
                        st.session_state.prompts[prompt_key] = prompt
                        st.success("Prompt erfolgreich generiert.")
                    except Exception as e:
                        st.error(f"Fehler bei OpenAI: {e}")

            if prompt_key in st.session_state.prompts:
                st.text_area("Prompt:", st.session_state.prompts[prompt_key], height=150)

                if st.button(f"🎨 Bild generieren {idx+1}"):
                    with st.spinner("Bild wird mit replicate erstellt..."):
                        try:
                            output_url = replicate_client.run(
                                "google/imagen-4",
                                input={
                                    "prompt": st.session_state.prompts[prompt_key],
                                    "aspect_ratio": "16:9",
                                    "safety_filter_level": "block_medium_and_above"
                                }
                            )
                            st.session_state.images[image_key] = output_url
                            st.success("Bild erfolgreich generiert.")
                        except Exception as e:
                            st.error(f"Fehler bei replicate: {e}")

            if image_key in st.session_state.images:
                st.image(st.session_state.images[image_key], caption="KI-Bild", use_container_width=True)
# %%