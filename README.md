# Your Wise Friend Journal

https://github.com/user-attachments/assets/137b8a3e-2df1-4c41-8317-ed16139fbd73

# Setup guide
## Create a virtual environment
`conda create --name wisefriend` `conda activate wisefriend`

## Install dependencies
* `conda install pip && cd wise-friend && pip install -r requirements.txt`
* precommit `pre-commit install`
* install postgres `conda install -y -c conda-forge postgresql`
* run the postgres docker container `docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16` (ensure to install docker)

## Start the application locally
* Setup `.streamlit/secrets.toml` (ask project owners about how to setup the secrets). Make sure you don't check in the secrets
* `sh start.sh`. For dry-run (avoid making external calls, which costs money), `sh start.sh dry-run`

# Debug
* To connect to the vectore store
`psql -h localhost -p 6024 -U langchain -d langchain` (password: langchain)
* To get all tables in the database
`\dt`
* To get table schema for a table
`\d {table_name};`