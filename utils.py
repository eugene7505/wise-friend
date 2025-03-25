import logging
from datetime import datetime

from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import SystemMessage

from langchain_fireworks import ChatFireworks, FireworksEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List

import config

# Calculate similarity score for citations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def retrieve(query: str, vector_store: VectorStore):
    retrieved_docs = vector_store.similarity_search(query)
    return retrieved_docs


def generate(query: str, context: List[Document], llm: ChatFireworks, prompt):
    docs_content = "\n\n".join(doc.page_content for doc in context)
    base_messages = prompt.invoke(
        {"question": query, "context": docs_content, "history": None}
    ).to_messages()
    # Prepend the supportive system message
    supportive_message = SystemMessage(
        content="You are a wise, supportive inner voice. Offer empathetic, gentle guidance using provided context to replace self-criticism with empowering insights or new perspectives."
    )

    messages = [supportive_message] + base_messages

    response = llm.invoke(messages)
    return response.content


def load_vector_stores(embeddings):
    wise_store = PGVector.from_existing_index(
        embedding=embeddings,
        connection=config.PSQL_URL,
        collection_name=config.WISE_COLLECTION,
    )
    journal_store = PGVector.from_existing_index(
        embedding=embeddings,
        connection=config.PSQL_URL,
        collection_name=config.JOURNAL_COLLECTION,
    )
    return wise_store, journal_store


def setup_models():
    llm = ChatFireworks(
        model=config.CHAT_MODEL,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    embeddings = FireworksEmbeddings(
        model=config.EMBEDDINGS_MODEL,
    )
    return llm, embeddings


def add_wise_entry(wise_store, file_path: str):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    # RecursiveCharacterTextSplitter allows you to split based on sentence boundaries,
    # and then split the sentences into chunks of a certain size, if the sentence is too long.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    batch_size = 200
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i : i + batch_size]
        wise_store.add_documents(batch)


def add_journal_entry(
    journal_store, entry: str, date=datetime.now().strftime("%Y-%m-%d")
):
    journal_store.add_texts([entry], metadatas=[{"date": date}])


def get_journal_entries_with_similar(
    journal_store,
    anchor: str,
    date=datetime.now().strftime("%Y-%m-%d"),
    threshold=0.3,
    k=5,
):
    entries = journal_store.similarity_search_with_score(
        anchor, k=k, filter={"date": {"$gte": date}}
    )
    return [entry for entry in entries if entry[1] <= threshold]


def get_journal_entries(
    journal_store, anchor: str, date=datetime.now().strftime("%Y-%m-%d"), k=5
):
    # todo
    pass


# Functions to display citations
def embedding_paragraph(paragrph: str, embeddings_model) -> np.ndarray:
    embedding = embeddings_model.embed_query(paragrph)
    return np.array(embedding).reshape(1, -1)


def calculate_semantic_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    # Calculate cosine similarity
    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity_score

def display_top_n_citations(context: List[Document], llm_response: str, embeddings: FireworksEmbeddings, n: int):
    llm_response_embedding = embedding_paragraph(llm_response, embeddings)

    similarity_scores = []
    for idx, doc in enumerate(context):
        doc_content = doc.page_content[:1000]
        doc_embedding = embedding_paragraph(doc_content, embeddings)
        similarity_score = calculate_semantic_similarity(llm_response_embedding, doc_embedding)
        similarity_scores.append([idx, similarity_score, doc_content])
        sorted_similarity_scores = sorted(similarity_scores, key= lambda x: x[1], reverse=True)
    top_n_citations = sorted_similarity_scores[:n]
    return [citation[2] for citation in top_n_citations]