# Description: Configuration file for the project

# Fireworks model description:
# https://fireworks.ai/models/fireworks/llama-v3p1-8b-instruct
CHAT_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
EMBEDDINGS_MODEL = "nomic-ai/nomic-embed-text-v1.5"
WISE_COLLECTION = "wise_friend"
JOURNAL_COLLECTION = "journal"
WISE_COLLECTION_TABLE = "collection_table"
PROMPT = "randomguy/rag_with_chat_history_and_citation"  # Original: "rlm/rag-prompt"
PSQL_URL = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!

TEST_USER_ID = "test123"
