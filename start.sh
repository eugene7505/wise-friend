export LANGSMITH_TRACING="true"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
if [[ "$1" == "dry-run" ]]; then
    streamlit run wise_friend_streamlit.py --server.port 8501 -- dry-run
elif [[ "$1" == "deploy" ]]; then
    nohup streamlit run wise_friend_streamlit.py --server.port 8501 &
else
    streamlit run wise_friend_streamlit.py --server.port 8501
fi
