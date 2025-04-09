import logging
import re
from datetime import datetime
from enum import Enum

import numpy as np
from langchain import hub
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.vectorstores import VectorStore
from langchain_fireworks import ChatFireworks, FireworksEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Calculate similarity score for citations
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text, insert, Table, MetaData, Sequence, Column, Integer, String
from typing_extensions import List

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


prompt = hub.pull(config.PROMPT)


class Category(Enum):
    DOCUMENT = "document"


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


def create_log_table(engine):
    # creates all other dependent tables if not exists
    metadata = MetaData()
    # Define a table using metadata
    log_table = Table(
        "log_table",
        metadata,
        Column("id", Integer, Sequence("some_id_seq", start=1), primary_key=True),
        Column("content", String, nullable=False),
        Column("date", String, nullable=False),
        Column("category", String, nullable=False),
    )
    metadata.create_all(engine)
    return log_table


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


def log_entry(log_table, content, date, category, engine):
    query = insert(log_table).values(content=content, date=date, category=category)
    with engine.connect() as conn:
        conn.execute(query)
        conn.commit()


def add_wise_entry(wise_store, file_path: str):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    # Clean citations like [1], [23], etc.
    for doc in docs:
        doc.page_content = re.sub(r"\[\d+\]", "", doc.page_content)

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


def get_journal_entries_with_similar(journal_store, anchor: str, threshold=0.3, k=5):
    entries = journal_store.similarity_search_with_score(anchor, k=k)
    return [
        (entry[0].page_content, entry[0].metadata["date"])
        for entry in entries
        if entry[1] <= threshold
    ]


def get_journal_entries(engine, k=5):
    with engine.connect() as connection:
        result = connection.execute(
            text(
            f"SELECT e.document, e.cmetadata->>'date' FROM langchain_pg_embedding e "
            f"JOIN langchain_pg_collection c "
            f"ON e.collection_id = c.uuid "
            f"WHERE c.name = '{config.JOURNAL_COLLECTION}' "
            f"ORDER BY e.cmetadata->>'date' DESC "
            f"LIMIT {k};"
            )
        ).fetchall()
    return result


def get_wise_documents(engine):
    with engine.connect() as connection:
        result = connection.execute(
            text(
                f"SELECT content, date FROM log_table where category = '{Category.DOCUMENT.value}'"
            )
        ).fetchall()
    # list of (content, date) tuples, filter out empty entries
    return [row for row in result if row[0]]


# Functions to display citations
def embedding_paragraph(paragrph: str, embeddings_model) -> np.ndarray:
    embedding = embeddings_model.embed_query(paragrph)
    return np.array(embedding).reshape(1, -1)


def calculate_semantic_similarity(
    embedding1: np.ndarray, embedding2: np.ndarray
) -> float:
    # Calculate cosine similarity
    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity_score


def display_top_n_citations(
    context: List[Document],
    llm_response: str,
    embeddings: FireworksEmbeddings,
    n: int,
):
    llm_response_embedding = embedding_paragraph(llm_response, embeddings)

    similarity_scores = []
    for idx, doc in enumerate(context):
        doc_content = doc.page_content[:1000]
        doc_embedding = embedding_paragraph(doc_content, embeddings)
        similarity_score = calculate_semantic_similarity(
            llm_response_embedding, doc_embedding
        )
        similarity_scores.append([idx, similarity_score, doc_content])
        sorted_similarity_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True
        )
    top_n_citations = sorted_similarity_scores[:n]
    return [citation[2] for citation in top_n_citations]
