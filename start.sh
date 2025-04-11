export LANGSMITH_TRACING="true"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_API_KEY="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"
export FIREWORKS_API_KEY="fw_3ZWqWSsKGFMvXM1qdAiTxXrw"
if [[ "$1" == "dry-run" ]]; then
    streamlit run wise_friend_streamlit.py --server.port 8506 -- dry-run
else
    streamlit run wise_friend_streamlit.py --server.port 8506
fi