import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Anime Recommender", layout="wide")


@st.cache_resource
def init_pipeline():
    return AnimeRecommendationPipeline()


pipeline = init_pipeline()

st.title("Anime Recommender")

query = st.text_input("Enter your favourite Anime")
if query:
    with st.spinner("Fetching recommendation for you .........."):
        response = pipeline.recommend(query)
        st.markdown("### Recommendations")
        st.write(response)
