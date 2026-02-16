import os
import io

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_bytes

from google import genai
from google.genai import types


load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)


def pdf_premiere_page_en_jpeg(uploaded_file) -> bytes:
    images = convert_from_bytes(uploaded_file.read())
    first_page: Image.Image = images[0]

    buf = io.BytesIO()
    first_page.save(buf, format="JPEG")
    return buf.getvalue()


def appeler_gemini_vision(prompt: str, job_description: str, image_bytes: bytes) -> str:
    if not job_description.strip():
        return (
            "Merci de coller une description de poste (JD) avant de lancer l'analyse."
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            "DESCRIPTION DU POSTE (JD) :\n" + job_description,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
    )
    return response.text or "(Réponse vide)"


PROMPT_REVIEW_FR = """
Tu es un(e) responsable RH technique expérimenté(e).

Analyse le CV (fourni en image) par rapport à la description de poste (JD).

Contraintes :
- Base-toi uniquement sur le CV (ne pas inventer)
- Si une information manque, dis-le clairement

Format :
1) Résumé du profil (5-7 lignes)
2) Points forts (bullet points)
3) Points faibles / manques (bullet points)
4) Recommandation globale (Oui / Mitigé / Non) + justification
""".strip()

PROMPT_ATS_FR = """
Tu es un système ATS (Applicant Tracking System) spécialisé dans les profils tech.

Compare le CV (fourni en image) avec la description de poste (JD).

Contraintes :
- Base-toi uniquement sur le CV (ne pas inventer)
- Si une information manque, dis-le clairement
- Le pourcentage doit être cohérent avec les arguments

Format :
- Pourcentage de correspondance : XX%
- Mots-clés / compétences manquants (liste)
- Mots-clés / compétences présents (liste courte)
- Conclusion (3-5 lignes) + 2 conseils d'amélioration du CV
""".strip()


st.set_page_config(page_title="ATS CV (Gemini) — Lab IA Générative", layout="centered")
st.title("📄 ATS CV — Analyse avec Gemini")
st.caption("JD + CV PDF → conversion en image → analyse par LLM multimodal.")

job_description = st.text_area(
    "🧾 Description du poste (JD) :",
    key="jd",
    height=220,
    placeholder="Colle ici l'offre d'emploi...",
)

uploaded_file = st.file_uploader("📎 Téléverse ton CV (PDF)", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    run_review = st.button("🧑‍💼 Analyse RH")
with col2:
    run_ats = st.button("🤖 Score ATS")

show_preview = st.checkbox("Afficher l'aperçu du CV", value=False)

if uploaded_file is None:
    st.warning("Téléverse un CV PDF pour commencer.")
else:
    st.success("CV PDF téléversé.")
    image_bytes = pdf_premiere_page_en_jpeg(uploaded_file)

    if show_preview:
        st.image(image_bytes, caption="1ère page du CV", use_container_width=True)

    if run_review:
        resultat = appeler_gemini_vision(PROMPT_REVIEW_FR, job_description, image_bytes)
        st.subheader("Résultat — Analyse RH")
        st.write(resultat)

    if run_ats:
        resultat = appeler_gemini_vision(PROMPT_ATS_FR, job_description, image_bytes)
        st.subheader("Résultat — ATS Match")
        st.write(resultat)
