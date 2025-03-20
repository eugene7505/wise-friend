import logging
from datetime import datetime

from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_fireworks import ChatFireworks, FireworksEmbeddings
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

    # Prepend the supportive system message
    supportive_message = SystemMessage(content = "You are a wise, supportive inner voice. Offer empathetic, gentle guidance to foster self-understanding and growth by using provided context from The Almanack of Naval Ravikant.")
    
    base_messages = HumanMessage(content = """After reading the question or journal entry, craft a response in a compassionate, supportive, and empowering tone, similar to how a therapist would communicate. Acknowledge the user's emotions, validate their experiences, and offer thoughtful insights or gentle guidance.

    Your response should be concise, well-structured, easily understandable, and provide a sense of encouragement. If the retrieved context contains multiple references, integrate them seamlessly while staying true to the original contexts in a warm and empathetic tone.

    If you don’t know the answer, express that honestly while still offering validation or encouragement. Remember, the goal is to provide a supportive and empowering response that fosters self-understanding and growth.
                                 
    **Length Constraint**: Ensure the response is less than 3 paragraphs or 150 words.

    Question: {question} 
    Chat History: {history}

    Context: {context} 
    Answer:""".format(question=query, history=None, context=docs_content)
    )
    
    messages = [supportive_message] + [base_messages]

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
        batch = all_splits[i:i + batch_size]
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
