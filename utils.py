import logging
from datetime import datetime

from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_fireworks import ChatFireworks, FireworksEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def retrieve(query: str, vector_store: VectorStore):
    retrieved_docs = vector_store.similarity_search(query)
    return retrieved_docs


def generate(query: str, context: List[Document], llm: ChatFireworks, prompt):
    docs_content = "\n\n".join(doc.page_content for doc in context)
    messages = prompt.invoke({"question": query, "context": docs_content})
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
    wise_store.add_documents(all_splits)


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


# if __name__ == "__main__":
#     llm, embeddings = setup_models()
#     # init wise_store and journal store
#     wise_store, journal_store = setup_vector_stores(embeddings)
#     prompt = hub.pull(config.PROMPT)

#     journal = input("How are you feeling today? \n")
#     past_journal = get_journal_entries_with_similar(journal_store, journal)
#     print(f"Retrieved past journal entries: {past_journal}")

#     retrieved_docs = retrieve(journal, wise_store)
#     print(f"Retrieved {len(retrieved_docs)} documents from the wise_store")
#     response = generate(journal, retrieved_docs, llm, prompt)
#     add_journal_entry(journal_store, journal)

#     print(response)
