from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.cutom_exception import CustomException

load_dotenv()

logger = get_logger(__name__)


def main():
    try:
        logger.info("Starting to build the pipeline")
        loader = AnimeDataLoader(
            "data/anime_with_synopsis.csv", "data/anime_updated.csv"
        )
        processed_data = loader.load_and_process()

        logger.info("Data loaded and processed")

        vector_builder = VectorStoreBuilder(processed_data)
        vector_builder.build_and_save_vectorstore()

        logger.info("Vector stored and building now ..........")

        logger.info("Pipline built Successfully")

    except Exception as e:
        logger.error(f"Failed to build {str(e)}")
        raise CustomException("Error during building", e)


if __name__ == "__main__":
    main()
