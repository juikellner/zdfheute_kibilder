import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import openai
import replicate

# 🔐 API-Schlüssel aus .env oder secrets.toml laden
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN") or st.secrets.get("REPLICATE_API_TOKEN")

# 📰 ZDF Top-Teaser scrapen
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9"
}

data = []
url = "https://www.zdf.de/nachrichten"

try:
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")
    picture_blocks = soup.find_all("picture", class_="slrzex8")

    for pic in picture_blocks[:3]:
        img_tag = pic.find("img")
        if not img_tag:
            continue

        srcset = img_tag.get("srcset", "")
        entries = [s.strip().split(" ")[0] for s in srcset.split(",")]
        image_url = entries[-1] if entries else img_tag.get("src")

        parent = pic.find_parent("section") or pic.find_parent("li") or pic.parent
        headline_tag = parent.find("a", class_="_nl_")
        dachzeile_tag = parent.find("span", class_="tsdggcs")

        if image_url and headline_tag:
            data.append({
                "image_url": image_url,
                "headline": headline_tag.get_text(strip=True),
                "dachzeile": dachzeile_tag.get_text(strip=True) if dachzeile_tag else "[Keine Dachzeile]",
                "url": "https://www.zdf.de" + headline_tag.get("href")
            })

except Exception as e:
    st.error(f"Fehler beim Scraping: {e}")

# 🧪 Fallback-Daten, wenn nichts gefunden
if not data:
    st.warning("Keine ZDF-Daten gefunden. Zeige Testdaten.")
    data = [{
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/e5/ZDF_logo_2021.svg",
        "headline": "Test: KI verändert die Welt",
        "dachzeile": "Technologie",
        "url": "https://www.zdf.de"
    }]

# 📺 UI Layout
st.set_page_config(page_title="ZDFheute KI-Bilder", layout="wide")
st.title("📰 ZDFheute KI-Bilder Generator")

for idx, item in enumerate(data):
    st.divider()
    cols = st.columns([1, 2])

    with cols[0]:
        try:
            img_response = requests.get(item["image_url"], timeout=10)
            image = Image.open(BytesIO(img_response.content))
            st.image(image, caption="Originalbild", use_container_width=True)
        except Exception as e:
            st.warning(f"Fehler beim Laden des Bildes: {e}")

    with cols[1]:
        st.markdown(f"### [{item['headline']}]({item['url']})")
        st.caption(item["dachzeile"])

        prompt_key = f"prompt_{idx}"
        image_key = f"image_{idx}"

        # 🎨 Prompt generieren
        with st.form(key=f"form_prompt_{idx}"):
            submitted = st.form_submit_button("🎨 Prompt generieren")
            if submitted:
                with st.spinner("Erzeuge Prompt..."):
                    try:
                        completion = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "Du bist ein Prompt-Experte für fotorealistische, kinoreife Bilder."},
                                {"role": "user", "content": f"Erstelle einen hochwertigen Bildprompt auf Basis von:\n\nDachzeile: {item['dachzeile']}\nSchlagzeile: {item['headline']}"}
                            ]
                        )
                        prompt = completion.choices[0].message["content"]
                        st.session_state[prompt_key] = prompt
                    except Exception as e:
                        st.session_state[prompt_key] = f"Fehler bei Prompt-Erstellung: {e}"

        if prompt_key in st.session_state:
            st.text_area("Prompt", st.session_state[prompt_key], height=200)

        # 🧠 KI-Bild generieren
        with st.form(key=f"form_image_{idx}"):
            clicked = st.form_submit_button("🧠 Bild generieren")
            if clicked and prompt_key in st.session_state:
                with st.spinner("Bild wird generiert..."):
                    try:
                        output = replicate.run(
                            "stability-ai/sdxl",
                            input={"prompt": st.session_state[prompt_key]}
                        )
                        st.session_state[image_key] = output[0] if isinstance(output, list) else output
                    except Exception as e:
                        st.session_state[image_key] = f"Fehler bei Bildgenerierung: {e}"

        if image_key in st.session_state:
            if isinstance(st.session_state[image_key], str) and st.session_state[image_key].startswith("http"):
                st.image(st.session_state[image_key], caption="KI-generiertes Bild", use_container_width=True)
            else:
                st.error(st.session_state[image_key])
