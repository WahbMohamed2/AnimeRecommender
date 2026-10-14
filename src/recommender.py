from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq

from src.prompt_template import get_anime_prompt


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str, max_tokens: int = 800):
        self.llm = ChatGroq(
            api_key=api_key, model=model_name, temperature=0, max_tokens=max_tokens
        )
        self.prompt = get_anime_prompt()
        self.retriever = retriever

        # LCEL replacement for langchain.chains.RetrievalQA, which was removed in
        # LangChain 1.0. Retrieval and the raw query are fanned out in parallel,
        # then `assign` adds the answer while keeping the source documents.
        self.chain = RunnableParallel(
            source_documents=self.retriever,
            question=RunnablePassthrough(),
        ).assign(
            answer=(
                {
                    "context": lambda x: format_docs(x["source_documents"]),
                    "question": lambda x: x["question"],
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        )

    def get_recommendation(self, query: str) -> str:
        return self.chain.invoke(query)["answer"]

    def get_recommendation_with_sources(self, query: str) -> dict:
        return self.chain.invoke(query)
