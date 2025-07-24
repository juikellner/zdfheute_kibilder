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

# Scrape top news articles from ZDFheute with best image resolution
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
            images = sorted(images, key=lambda x: int(x.split("~")[-1].split("x")[0]) if "~" in x else 0, reverse=True)
            img_url = images[0] if images else img.get("src")

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
def generate_prompt(headline, dachzeile, image_url):
    try:
        vision_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Du bist ein kreativer Prompt-Designer für Text-zu-Bild-KI. Beschreibe den visuellen Inhalt dieses Bildes in stichpunktartiger Form für einen Prompt."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": f"Bitte analysiere das Bild."}
                    ]
                }
            ],
            max_tokens=1000
        )
        image_description = vision_response.choices[0].message.content.strip()

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein kreativer Prompt-Designer für Text-zu-Bild-KI."},
                {"role": "user", "content": f"Erstelle einen filmisch-realistischen Bildprompt auf Englisch für folgende ZDF-Schlagzeile: '{headline}'\nDachzeile: '{dachzeile}'\nNutze außerdem diese Bildbeschreibung: {image_description}. Der Prompt soll für ein Modell wie 'ideogram-ai/ideogram-v3-turbo' geeignet sein."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Fehler bei Prompt-Erstellung: {e}")
        return None

# Generate image with Replicate (ideogram-v3-turbo)
def generate_image_url(prompt):
    try:
        os.environ["REPLICATE_API_TOKEN"] = replicate_token
        output = replicate.run(
            "ideogram-ai/ideogram-v3-turbo",
            input={"prompt": prompt, "aspect_ratio": "3:2"}
        )

        result = output[0] if isinstance(output, list) and len(output) > 0 else output
        image_url = str(result)  # sicherstellen, dass es ein String ist
        return image_url
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
            st.session_state[f"generated_{idx}"] = {"prompt": None, "image_url": None}

        st.image(item["image_url"], caption="Originalbild", width=800)

        if st.button(f"✨ Prompt & Bild generieren für: {item['headline']}", key=f"btn_generate_{idx}"):
            with st.spinner("🔍 Erzeuge Prompt..."):
                prompt = generate_prompt(item['headline'], item['dachzeile'], item['image_url'])
                st.session_state[f"generated_{idx}"]["prompt"] = prompt

            if prompt:
                st.markdown("**📝 Generierter Prompt:**")
                st.markdown(f"<code style='font-size: 0.75rem; word-wrap: break-word; white-space: pre-wrap;'>{prompt}</code>", unsafe_allow_html=True)

                with st.spinner("🎨 Erzeuge KI-Bild..."):
                    image_url = generate_image_url(prompt)
                    st.session_state[f"generated_{idx}"]["image_url"] = image_url
            st.rerun()

        generated = st.session_state.get(f"generated_{idx}", {})
        prompt = generated.get("prompt")
        image_url = generated.get("image_url")

        if prompt:
            st.markdown("**📝 Generierter Prompt:**")
            st.markdown(f"<code style='font-size: 0.75rem; word-wrap: break-word; white-space: pre-wrap;'>{prompt}</code>", unsafe_allow_html=True)

        if image_url:
            col1, col2 = st.columns(2)
            with col1:
                st.image(item["image_url"], caption="Originalbild", width=400)
            with col2:
                st.image(image_url, caption="KI-generiertes Bild", width=400)
        elif prompt:
            st.info("⚠️ Kein KI-Bildlink von Replicate erhalten.")
else:
    st.warning("Keine Daten gefunden.")
