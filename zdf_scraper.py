import streamlit as st
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import os
from dotenv import load_dotenv
import openai
import replicate
import re
import base64

# Load API keys from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN")
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

# Streamlit app title
st.set_page_config(layout="wide")
st.title("📰 ZDFheute KI-Teaser")

# Hinweistext (klein und responsiv)
st.markdown("<p style='font-size: 1.1rem; line-height: 1.4;'>🔍 Diese Anwendung scrapt die drei Top-Teaser auf zdfheute.de und nutzt die <code>GPT-4o/4-Modelle</code> von OpenAI zur Bildbeschreibung und Prompt-Erstellung. Der Prompt zum Erstellen des KI-Bildes wird mit den Informationen aus Bildbeschreibung, Bildquelle (URL), Schlag- und Dachzeile generiert. Für die Bildgenerierung wird das <code>imagen-4-fast-Modell</code> von Google auf replicate.com eingesetzt.</p>", unsafe_allow_html=True)

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
                title = a_tag.get_text(strip=True) if a_tag else "Kein Titel gefunden"
                article_url = "https://www.zdfheute.de" + a_tag["href"] if a_tag and a_tag.has_attr("href") else ""

                # Suche nach h2 mit Dach- und Schlagzeile
                h2_tag = parent.find("h2", class_="hvzzigm") or parent.find("h2", class_="h1npplxp")

            if h2_tag:
                # Dachzeile suchen (egal ob tsdggcs + o1ximh7k oder andere)
                dach_span = h2_tag.find("span", class_="o1ximh7k tsdggcs") or h2_tag.find("span", class_="of88x80 tsdggcs")
                dachzeile = dach_span.get_text(strip=True) if dach_span else ""

                # Schlagzeile suchen
                headline_a = h2_tag.find("a")
                title = headline_a.get_text(strip=True) if headline_a else "Kein Titel gefunden"
                article_url = "https://www.zdfheute.de" + headline_a["href"] if headline_a and headline_a.has_attr("href") else ""
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

# Bildbeschreibung mit Qwen-VL über Fireworks.ai

def llama_image_description(image_url, context_from_url):
    try:
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_bytes = image_response.content
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {fireworks_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "qwen/qwen2-72b-vl-chat",
            "messages": [
                {"role": "system", "content": "Du bist ein Nachrichtenmodell, das journalistische Bilder in Stichpunkten beschreibt. Beschreibe sachlich und kurz auf Deutsch."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Kontext aus URL: {context_from_url}. Beschreibe den Inhalt des Bildes in 3–5 Stichpunkten."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.2
        }

        response = requests.post(
            "https://api.fireworks.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()

    except Exception as e:
        st.warning(f"Qwen-Bildbeschreibung (Fireworks) fehlgeschlagen: {e}")
        return ""

def generate_prompt(headline, dachzeile, image_url):
    try:
        context_from_url = extract_context_from_url(image_url)
        image_description = llama_image_description(image_url, context_from_url)

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein kreativer Prompt-Designer für Text-zu-Bild-KI im Nachrichten-Bereich."},
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