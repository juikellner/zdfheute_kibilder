import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import openai
import replicate

# 👉 Umgebungsvariablen laden
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN") or st.secrets.get("REPLICATE_API_TOKEN")

# 👉 API-Keys setzen
openai.api_key = openai_api_key
os.environ["REPLICATE_API_TOKEN"] = replicate_token

# 👉 ZDF-Daten abrufen
url = "https://www.zdf.de/nachrichten"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

data = []
try:
    response = requests.get(url, headers=headers, timeout=10)
    st.write(f"ZDF Statuscode: {response.status_code}")
    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.find_all("article", class_="news-teaser-item")
    st.write(f"Gefundene Artikel: {len(articles)}")

    for article in articles[:3]:
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

except Exception as e:
    st.error(f"Fehler beim Abrufen der ZDF-Seite: {e}")

# 👉 Fallback bei leerem Ergebnis
if not data:
    st.warning("Konnte keine Live-Daten von ZDF abrufen. Zeige stattdessen Testdaten.")
    data = [
        {
            "image_url": "https://via.placeholder.com/600x300.png?text=ZDF-Testbild",
            "headline": "Test-Schlagzeile: KI verändert alles",
            "dachzeile": "Technologie"
        }
    ]

# 👉 Streamlit UI
st.set_page_config(page_title="ZDFheute KI-Bilder", layout="wide")
st.title("📰 ZDFheute KI-Bilder Generator")

for idx, entry in enumerate(data):
    st.divider()
    cols = st.columns([1, 2])

    with cols[0]:
        try:
            img_response = requests.get(entry["image_url"], headers=headers, timeout=10)
            if img_response.status_code == 200:
                image = Image.open(BytesIO(img_response.content))
                st.image(image, caption="Originalbild", use_container_width=True)
            else:
                st.warning(f"Bild konnte nicht geladen werden (Statuscode: {img_response.status_code})")
        except Exception as e:
            st.error(f"Fehler beim Laden des Bildes: {e}")

    with cols[1]:
        st.subheader(entry["headline"])
        st.caption(entry["dachzeile"])

        prompt_key = f"prompt_{idx}"
        image_key = f"image_{idx}"

        # 👉 Prompt-Generierung
        with st.form(key=f"form_prompt_{idx}"):
            submitted_prompt = st.form_submit_button("🎨 Prompt generieren")
            if submitted_prompt:
                with st.spinner("Erstelle Prompt..."):
                    try:
                        completion = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "Du bist ein Prompt-Experte für fotorealistische, kinoreife Bilder."},
                                {"role": "user", "content": f"Erstelle einen hochwertigen Replicate-Prompt auf Basis von:\n\nDachzeile: {entry['dachzeile']}\nSchlagzeile: {entry['headline']}"}
                            ]
                        )
                        prompt = completion.choices[0].message["content"]
                        st.session_state[prompt_key] = prompt
                    except Exception as e:
                        st.session_state[prompt_key] = f"Fehler bei Prompt-Erstellung: {e}"

        if prompt_key in st.session_state:
            st.text_area("Prompt", st.session_state[prompt_key], height=200)

        # 👉 Bildgenerierung
        with st.form(key=f"form_image_{idx}"):
            submitted_image = st.form_submit_button("🧠 Bild generieren")
            if submitted_image and prompt_key in st.session_state:
                with st.spinner("Generiere Bild über Replicate..."):
                    try:
                        output = replicate.run(
                            "stability-ai/sdxl",  # Alternativ: "cjwbw/anything-v4.0" o. ä.
                            input={"prompt": st.session_state[prompt_key]}
                        )
                        st.session_state[image_key] = output[0] if isinstance(output, list) else output
                    except Exception as e:
                        st.session_state[image_key] = f"Fehler: {e}"

        # 👉 Anzeige des KI-generierten Bildes
        if image_key in st.session_state:
            if isinstance(st.session_state[image_key], str) and st.session_state[image_key].startswith("http"):
                st.image(st.session_state[image_key], caption="KI-generiertes Bild", use_container_width=True)
            else:
                st.error(st.session_state[image_key])
