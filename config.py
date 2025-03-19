import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["FIREWORKS_API_KEY"] = "fw_3ZWqWSsKGFMvXM1qdAiTxXrw"

CHAT_MODEL = "accounts/fireworks/models/llama-v3p2-3b-instruct"
EMBEDDINGS_MODEL = "nomic-ai/nomic-embed-text-v1.5"
WISE_DOC = "/Users/yichinw/Desktop/The-Almanack-of-Naval-Ravikant_Final.pdf"
WISE_COLLECTION = "wise_friend"
JOURNAL_COLLECTION = "journal"
PROMPT = "rlm/rag-prompt"
PSQL_URL = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
