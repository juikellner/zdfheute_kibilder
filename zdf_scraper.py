import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import openai
import replicate

# Load API keys from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN")

# Streamlit app title
st.set_page_config(layout="wide")
st.title("📰 ZDFheute KI-Bilder Generator")

# Scrape top news articles from ZDFheute
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
            # Get largest image from srcset
            img = pic.find("img")
            if not img:
                continue
            srcset = img.get("srcset", "")
            largest_img = sorted(
                [s.split(" ")[0] for s in srcset.split(",") if s.strip()],
                key=lambda x: int(x.split("~")[-1].split("x")[0]) if "~" in x else 0,
                reverse=True
            )
            img_url = largest_img[0] if largest_img else img.get("src")

            # Try to get title and link
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

            results.append({
                "image_url": img_url,
                "headline": title,
                "dachzeile": dachzeile,
                "url": article_url
            })
        return results
    except Exception as e:
        st.error(f"Fehler beim Scraping: {e}")
        return []

# Generate image prompt using OpenAI
def generate_prompt(headline, dachzeile):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein kreativer Prompt-Designer für Text-zu-Bild-KI."},
                {"role": "user", "content": f"Erstelle einen filmisch-realistischen Bildprompt für folgende ZDF-Schlagzeile: '{headline}'\nDachzeile: '{dachzeile}'\nDer Prompt soll für ein fotorealistisches Text-zu-Bild-Modell wie 'bytedance/seedream-3' geeignet sein. Schreibe den Prompt auf Englisch."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Fehler bei Prompt-Erstellung: {e}")
        return None

# Generate image with Replicate (seedream-3)
def generate_image(prompt):
    try:
        os.environ["REPLICATE_API_TOKEN"] = replicate_token
        output = replicate.run(
            "bytedance/seedream-3",
            input={"prompt": prompt}
        )
        st.write("Replicate-Ausgabe:", output)  # Debug-Ausgabe anzeigen

        if isinstance(output, str) and output.startswith("http"):
            img_response = requests.get(output, timeout=20)
            if img_response.status_code == 200:
                return BytesIO(img_response.content)
            else:
                st.warning("Bild konnte nicht geladen werden.")
                return None
        else:
            st.warning("Ausgabe von Replicate ist leer oder ungültig.")
            return None
    except Exception as e:
        st.error(f"Fehler bei Bildgenerierung: {e}")
        return None

# MAIN APP FLOW
data = scrape_top_articles()

if data:
    for idx, item in enumerate(data):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(item["image_url"], use_container_width=True)
        with col2:
            st.markdown(f"### {item['headline']}")
            st.markdown(f"**{item['dachzeile']}**")
            st.markdown(f"🔗 [Zum Artikel]({item['url']})")

            if st.button(f"✨ Prompt & Bild generieren für: {item['headline']}", key=f"btn_generate_{idx}"):
                with st.spinner("Erzeuge Prompt und Bild..."):
                    prompt = generate_prompt(item['headline'], item['dachzeile'])
                    if prompt:
                        st.markdown("**📝 Generierter Prompt:**")
                        st.code(prompt)
                        image_data = generate_image(prompt)
                        if image_data:
                            st.image(image_data, caption="KI-generiertes Bild", use_container_width=True)
                        else:
                            st.error("❌ Bild konnte nicht generiert werden.")
                    else:
                        st.error("❌ Prompt konnte nicht erzeugt werden.")
else:
    st.warning("Keine Daten gefunden.")
