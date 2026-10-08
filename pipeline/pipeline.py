from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.cutom_exception import CustomException

logger = get_logger(__name__)


class AnimeRecommendationPipeline:
    def __init__(self, persist_dir="chroma_db"):
        try:
            logger.info("Initializing Recommendation Pipeline")

            vector_build = VectorStoreBuilder(persist_dir=persist_dir)

            retriever = vector_build.load_vector_store().as_retriever()

            self.recommender = AnimeRecommender(retriever, GROQ_API_KEY, MODEL_NAME)

            logger.info("Pipeline intialization is done .....")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline {str(e)}")
            raise CustomException("Error during Pipeline initialization", e)

    def recommend(self, query: str) -> str:
        try:
            logger.info(f"Recived a query {query}")

            recommendation = self.recommender.get_recommendation(query)
            logger.info("Recommendation generated succefuly.........")

            return recommendation

        except Exception as e:
            logger.error(f"Failed to recommend {str(e)}")
            raise CustomException("Error during recommend", e)
