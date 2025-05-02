import logging
import re

import numpy as np
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_fireworks import ChatFireworks, FireworksEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import supabase

# Import Langsmith for user feedback collection
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree


# Calculate similarity score for citations
from sklearn.metrics.pairwise import cosine_similarity
from typing_extensions import List
from typing import AsyncGenerator, Tuple
import uuid


import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def retrieve(query: str, vector_store: VectorStore, userid: str, dry_run: bool):
    retrieved_docs = (
        vector_store.similarity_search(query, filter={"userid": userid})
        if not dry_run
        else []
    )
    return retrieved_docs


# Helper function to concatenate outputs of the LLM
# for cleaner LangSmith tracing.
def concatenate(outputs: list):
    return "".join(x[0] for x in outputs if x and x[0])


# Generate streaming response
@traceable(reduce_fn=concatenate)
async def generate_streaming(
    query: str, context: List[Document], llm: ChatFireworks
) -> AsyncGenerator[Tuple[str, str], None]:
    docs_content = "\n\n".join(doc.page_content for doc in context)

    # Structure the prompt messages
    messages = [
        SystemMessage(content=config.SUPPORTIVE_MESSAGE_CONTENT),
        HumanMessage(
            content=config.PROMPT_TEMPLATE.format(
                journal_entry=query,
                response_starter_history=None,
                context=docs_content,
            )
        ),
    ]
    # Get the run.id to use it in logging user feedback for LLM responses
    # and for tracing.
    run = get_current_run_tree()
    run_id = run.id

    first_chunk = True
    async for chunk in llm.astream(messages):
        content = chunk.content or ""
        if first_chunk:
            yield content, run_id  # yield run_id with the first chunk
            first_chunk = False
        else:
            yield content, None  # only content afterwards


# Generate non-streaming response
@traceable
def generate(query: str, context: List[Document], llm: ChatFireworks):
    docs_content = "\n\n".join(doc.page_content for doc in context)

    # Structure the prompt messages
    messages = [
        SystemMessage(content=config.SUPPORTIVE_MESSAGE_CONTENT),
        HumanMessage(
            content=config.PROMPT_TEMPLATE.format(
                journal_entry=query,
                response_starter_history=None,
                context=docs_content,
            )
        ),
    ]

    run = get_current_run_tree()
    print(f"generate Run Id: {run.id}")
    response = llm.invoke(messages)
    return response.content, run.id


def load_vector_stores(client, embeddings):
    wise_store = SupabaseVectorStore(
        embedding=embeddings,
        client=client,
        table_name="wise_embeddings",
        query_name="similar_wise_documents",
    )
    journal_store = SupabaseVectorStore(
        embedding=embeddings,
        client=client,
        table_name="journal_embeddings",
        query_name=None,  # None, as we don't do similarity search for journal entries
    )
    return wise_store, journal_store


# def get_wise_collection_table(engine):
#     # creates all other dependent tables if not exists
#     metadata = MetaData()
#     # Define a table using metadata
#     table = Table(
#         config.WISE_COLLECTION_TABLE,
#         metadata,
#         Column("id", Integer, Sequence("some_id_seq", start=1), primary_key=True),
#         Column("userid", String, nullable=False),
#         Column("content", String, nullable=False),
#         Column("date", String, nullable=False),
#     )
#     metadata.create_all(engine)
#     return table


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


def add_wise_collection(
    client: supabase.Client, userid: str, content: str, date: str
) -> List:
    result = (
        client.table(config.WISE_COLLECTION_TABLE)
        .insert(
            {
                "id": str(uuid.uuid4()),
                "userid": userid,
                "content": content,
                "date": date,
            }
        )
        .execute()
    )
    return result.data


def add_wise_entry(
    wise_store: SupabaseVectorStore, file_path: str, userid: str, dry_run: bool = False
) -> None:
    if not dry_run:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        # Clean citations like [1], [23], etc.
        for doc in docs:
            doc.page_content = re.sub(r"\[\d+\]", "", doc.page_content)
            doc.metadata["userid"] = userid

        # RecursiveCharacterTextSplitter allows you to split based on sentence boundaries,
        # and then split the sentences into chunks of a certain size, if the sentence is too long.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)
        batch_size = 200
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i : i + batch_size]
            wise_store.add_documents(documents=batch)


def add_journal_entry(
    journal_store: VectorStore,
    entry: str,
    userid: str,
    date: str,
    dry_run: bool = False,
) -> None:
    if not dry_run:
        journal_store.add_texts(
            texts=[entry], metadatas=[{"userid": userid, "date": date}]
        )


def get_journal_entries(client: supabase.Client, userid: str, k=5) -> List:
    result = (
        client.table(config.JOURNAL_EMBEDDING_TABLE)
        .select("content, metadata->>date")
        .eq("metadata->>userid", userid)
        .order("metadata->>date", desc=True)
        .limit(k)
        .execute()
    )
    return result.data


def get_wise_collection(client: supabase.Client, userid: str) -> List:
    result = (
        client.table(config.WISE_COLLECTION_TABLE)  # replace with your table name
        .select("content, date")
        .eq("userid", userid)
        .execute()
    )
    return result.data


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
