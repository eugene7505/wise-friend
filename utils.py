import logging
import re
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

# Import Langsmith for user feedback collection
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree


# Calculate similarity score for citations
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text, insert, Table, MetaData, Sequence, Column, Integer, String
from typing_extensions import List
from typing import AsyncGenerator, Tuple


import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


prompt = hub.pull(config.PROMPT)


class Category(Enum):
    DOCUMENT = "document"


def retrieve(query: str, vector_store: VectorStore, dry_run: bool):
    retrieved_docs = vector_store.similarity_search(query) if not dry_run else []
    return retrieved_docs


def concatenate(outputs: list):
    return "".join(x[0] for x in outputs if x and x[0])


@traceable(reduce_fn=concatenate)
async def generate_streaming(
    query: str,
    context: List[Document],
    llm: ChatFireworks,
    prompt,
) -> AsyncGenerator[Tuple[str, str], None]:
    docs_content = "\n\n".join(doc.page_content for doc in context)
    base_messages = prompt.invoke(
        {"question": query, "context": docs_content, "history": None}
    ).to_messages()
    # Prepend the supportive system message
    supportive_message = SystemMessage(
        content="You are a wise, supportive inner voice. Offer empathetic, gentle guidance using provided context to replace self-criticism with empowering insights or new perspectives."
    )

    messages = [supportive_message] + base_messages
    run = get_current_run_tree()
    run_id = run.id
    first = True

    async for chunk in llm.astream(messages):
        content = chunk.content or ""
        if first:
            yield content, run_id  # yield run_id with the first chunk
            first = False
        else:
            yield content, None  # only content afterwards


@traceable
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
    run = get_current_run_tree()
    print(f"generate Run Id: {run.id}")
    response = llm.invoke(messages)
    return response.content, run.id


def load_vector_stores(embeddings, userid):
    wise_store = PGVector.from_existing_index(
        embedding=embeddings,
        connection=config.PSQL_URL,
        collection_name=f"{config.WISE_COLLECTION}_{userid}",
    )
    journal_store = PGVector.from_existing_index(
        embedding=embeddings,
        connection=config.PSQL_URL,
        collection_name=f"{config.JOURNAL_COLLECTION}_{userid}",
    )
    return wise_store, journal_store


def get_collection_table(engine):
    # creates all other dependent tables if not exists
    metadata = MetaData()
    # Define a table using metadata
    table = Table(
        config.WISE_COLLECTION_TABLE,
        metadata,
        Column("id", Integer, Sequence("some_id_seq", start=1), primary_key=True),
        Column("userid", String, nullable=False),
        Column("content", String, nullable=False),
        Column("date", String, nullable=False),
    )
    metadata.create_all(engine)
    return table


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


def update_collection(table, userid, content, date, engine):
    query = insert(table).values(userid=userid, content=content, date=date)
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


def add_journal_entry(journal_store, entry: str, date: str, dry_run: bool):
    if not dry_run:
        journal_store.add_texts([entry], metadatas=[{"date": date}])


def get_journal_entries_with_similar(journal_store, anchor: str, threshold=0.3, k=5):
    entries = journal_store.similarity_search_with_score(anchor, k=k)
    return [
        (entry[0].page_content, entry[0].metadata["date"])
        for entry in entries
        if entry[1] <= threshold
    ]


def get_journal_entries(engine, userid, k=5):
    collection_name = f"{config.JOURNAL_COLLECTION}_{userid}"
    with engine.connect() as connection:
        result = connection.execute(
            text(
                "SELECT e.document, e.cmetadata->>'date' FROM langchain_pg_embedding e "
                "JOIN langchain_pg_collection c "
                "ON e.collection_id = c.uuid "
                "WHERE c.name = :collection_name "
                "ORDER BY e.cmetadata->>'date' DESC "
                "LIMIT :limit;"
            ),
            {"collection_name": collection_name, "limit": k},
        ).fetchall()
    return result


def get_wise_documents(engine, userid):
    with engine.connect() as connection:
        result = connection.execute(
            text(
                f"SELECT content, date FROM {config.WISE_COLLECTION_TABLE} where userid = {userid}"
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
