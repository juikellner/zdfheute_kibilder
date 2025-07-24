import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import openai
import replicate

# Load .env for API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

st.set_page_config(page_title="ZDFheute KI-Bilder Generator", layout="wide")
st.title("📰 ZDFheute KI-Bilder Generator")

@st.cache_data(show_spinner=False)
def scrape_top_articles():
    url = "https://www.zdfheute.de"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    articles = []
    seen_titles = set()
    for pic in soup.find_all("picture", class_="slrzex8"):
        img_tag = pic.find("img")
        if not img_tag:
            continue

        image_url = img_tag.get("src")
        alt_text = img_tag.get("alt", "")
        parent = pic.find_parent("li") or pic.find_parent("div")
        link_tag = parent.find("a") if parent else None
        headline = link_tag.get_text(strip=True) if link_tag else alt_text
        link = link_tag.get("href") if link_tag and link_tag.get("href") else ""
        full_url = link if link.startswith("http") else f"{url}{link}"

        if headline not in seen_titles:
            articles.append({
                "image_url": image_url,
                "headline": headline,
                "subheadline": alt_text,
                "url": full_url
            })
            seen_titles.add(headline)

        if len(articles) >= 3:
            break

    return articles

def generate_prompt(headline, subheadline):
    try:
        system_prompt = "Erzeuge einen bildgenerierenden Prompt (für ein Text-zu-Bild Modell), der die folgende ZDF-Schlagzeile und Dachzeile kreativ, anschaulich, aber realistisch in ein Standbild umsetzt."
        user_input = f"Schlagzeile: {headline}\nDachzeile: {subheadline}"

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Fehler bei Prompt-Erstellung: {e}"

def generate_image(prompt):
    try:
        output = replicate.run(
            "google/imagen-4-fast",
            input={
                "prompt": prompt,
                "aspect_ratio": "4:3",
                "output_format": "jpg",
                "safety_filter_level": "block_only_high"
            }
        )
        return output.url() if hasattr(output, "url") else output
    except Exception as e:
        return f"Fehler bei Bildgenerierung: {e}"

articles = scrape_top_articles()

for idx, article in enumerate(articles):
    st.markdown("---")
    cols = st.columns([1, 2])

    with cols[0]:
        try:
            st.image(article["image_url"], caption="Originalbild", use_column_width=True)
        except:
            st.warning("Bild konnte nicht geladen werden.")

    with cols[1]:
        st.subheader(article["headline"])
        st.caption(article["subheadline"])
        st.markdown(f"[🔗 Zum Artikel]({article['url']})")

        prompt_key = f"prompt_{idx}"
        image_key = f"image_{idx}"

        if st.button("📝 Prompt generieren", key=f"btn_prompt_{idx}"):
            with st.spinner("Erzeuge Prompt mit GPT-4..."):
                prompt = generate_prompt(article["headline"], article["subheadline"])
                st.session_state[prompt_key] = prompt

        if prompt_key in st.session_state:
            st.text_area("🎯 KI-Prompt", st.session_state[prompt_key], height=150)

        with st.form(key=f"form_image_{idx}"):
            if st.form_submit_button("🎨 Bild generieren"):
                with st.spinner("Erzeuge Bild mit Imagen 4 Fast..."):
                    image_url = generate_image(st.session_state.get(prompt_key, ""))
                    st.session_state[image_key] = image_url

        if image_key in st.session_state:
            if isinstance(st.session_state[image_key], str) and st.session_state[image_key].startswith("http"):
                st.image(st.session_state[image_key], caption="KI-generiertes Bild", use_column_width=True)
            else:
                st.error(st.session_state[image_key])
                