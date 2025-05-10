# Your Wise Friend Journal

# Setup guide
## Create a virtual environment
`conda create --name wisefriend` `conda activate wisefriend`

## Install dependencies
* `conda install pip && cd wise-friend && pip install -r requirements.txt`
* precommit `pre-commit install`
* install postgres `conda install -y -c conda-forge postgresql`
* run the postgres docker container `docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16` (ensure to install docker)

## Start the application
* Setup `.streamlit/secrets.toml` (ask project owners about how to setup the secrets). Make sure you don't check in the secrets
* `sh start.sh`
    - For dry-run (avoid making external calls, which costs money), `sh start.sh dry-run`
    - For local deployment, `sh start.sh`
    - For prod deployment, `sh start.sh deploy` to deploy in the background

# Debug
* supabase https://supabase.com/dashboard/project/ddnhcucntwfwqmetvkwp
