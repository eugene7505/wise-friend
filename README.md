# Your Wise Friend Journal

https://github.com/user-attachments/assets/137b8a3e-2df1-4c41-8317-ed16139fbd73

# Setup guide
## Create a virtual environment
`conda create --name wisefriend`
`conda activate wisefriend`

## Setup environment variables
```
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="https://api.smith.langchain.com"
export LANGSMITH_API_KEY="..."
export FIREWORKS_API_KEY="..."
```

## Install dependencies
* `conda install pip`
* python dependencies `pip install -r requirements.txt`
* install postgres `conda install -y -c conda-forge postgresql`
* run the postgres docker container `docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16` (ensure to install docker)

## Start the application
* `streamlit run wise_friend_streamlit.py`
* To start in a dry-run mode without making the external LLM/embedding calls (save money), `streamlit run wise_friend_streamlit.py -- dry-run`

# Debug
* To connect to the vectore store
`psql -h localhost -p 6024 -U langchain -d langchain` (password: langchain)
