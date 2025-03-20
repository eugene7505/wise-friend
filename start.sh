export LANGSMITH_TRACING="true"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_API_KEY="your_langsmith_api_key" # make sure you don't check in your api key!
export FIREWORKS_API_KEY="your_fireworks_api_key" # make sure you don't check in your api key!

if [[ "$1" == "dry-run" ]]; then
    streamlit run wise_friend_streamlit.py -- dry-run
else
    streamlit run wise_friend_streamlit.py
fi