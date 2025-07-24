import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import openai
import replicate

# Umgebungsvariablen laden
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN") or st.secrets.get("REPLICATE_API_TOKEN")

# API-Keys setzen
openai.api_key = openai_api_key
os.environ["REPLICATE_API_TOKEN"] = replicate_token

# Datenstruktur
data = []

# Scraping
url = "https://www.zdf.de/nachrichten"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
try:
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")

    # 1. Einstiegsmodul
    einstiegsmodul = soup.find("section", {"data-testid": "einstiegsmodul"})
    if einstiegsmodul:
        headline = einstiegsmodul.select_one("span.tsdggcs")
        title = einstiegsmodul.select_one("a._nl_")
        image = einstiegsmodul.select_one("img")
        if headline and title and image:
            data.append({
                "headline": title.get_text(strip=True),
                "dachzeile": headline.get_text(strip=True),
                "image_url": image.get("src"),
                "url": "https://www.zdf.de" + title.get("href")
            })

    # 2. Zwei weitere Teaser
    top_teasers = soup.find_all("li", {"data-testid": "l-teaser"})
    for teaser in top_teasers[:2]:
        image = teaser.select_one("img")
        headline = teaser.select_one("a._nl_")
        dachzeile = teaser.select_one("span.tsdggcs")
        if image and headline and dachzeile:
            data.append({
                "headline": headline.get_text(strip=True),
                "dachzeile": dachzeile.get_text(strip=True),
                "image_url": image.get("src"),
                "url": "https://www.zdf.de" + headline.get("href")
            })

except Exception as e:
    st.error(f"Fehler beim Abrufen der ZDF-Daten: {e}")

# Fallback-Testdaten
if not data:
    st.warning("Keine ZDF-Daten gefunden – zeige Testartikel.")
    data = [
        {
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/e5/ZDF_logo_2021.svg",
            "headline": "Test: KI revolutioniert den Alltag",
            "dachzeile": "Technologie",
            "url": "https://www.zdf.de"
        }
    ]

# UI
st.set_page_config(page_title="ZDFheute KI-Bilder", layout="wide")
st.title("📰 ZDFheute KI-Bilder Generator")

for idx, entry in enumerate(data):
    st.divider()
    cols = st.columns([1, 2])

    with cols[0]:
        try:
            img_response = requests.get(entry["image_url"], timeout=10)
            image = Image.open(BytesIO(img_response.content))
            st.image(image, caption="Originalbild", use_container_width=True)
        except Exception as e:
            st.warning(f"Bildfehler: {e}")

    with cols[1]:
        st.markdown(f"### [{entry['headline']}]({entry['url']})")
        st.caption(entry['dachzeile'])

        prompt_key = f"prompt_{idx}"
        image_key = f"image_{idx}"

        with st.form(key=f"form_prompt_{idx}"):
            submitted_prompt = st.form_submit_button("🎨 Prompt generieren")
            if submitted_prompt:
                with st.spinner("Erstelle Prompt..."):
                    try:
                        completion = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "Du bist ein Prompt-Experte für fotorealistische, kinoreife Bilder."},
                                {"role": "user", "content": f"Erstelle einen hochwertigen Bildprompt auf Basis von:\n\nDachzeile: {entry['dachzeile']}\nSchlagzeile: {entry['headline']}"}
                            ]
                        )
                        prompt = completion.choices[0].message["content"]
                        st.session_state[prompt_key] = prompt
                    except Exception as e:
                        st.session_state[prompt_key] = f"Fehler: {e}"

        if prompt_key in st.session_state:
            st.text_area("Prompt", st.session_state[prompt_key], height=200)

        with st.form(key=f"form_image_{idx}"):
            submitted_image = st.form_submit_button("🧠 Bild generieren")
            if submitted_image and prompt_key in st.session_state:
                with st.spinner("Bild wird generiert..."):
                    try:
                        output = replicate.run(
                            "stability-ai/sdxl",
                            input={"prompt": st.session_state[prompt_key]}
                        )
                        image_url = output[0] if isinstance(output, list) else output
                        st.session_state[image_key] = image_url
                    except Exception as e:
                        st.session_state[image_key] = f"Fehler bei Bildgenerierung: {e}"

        if image_key in st.session_state:
            if isinstance(st.session_state[image_key], str) and st.session_state[image_key].startswith("http"):
                st.image(st.session_state[image_key], caption="KI-generiertes Bild", use_container_width=True)
            else:
                st.error(st.session_state[image_key])
