# Your Wise Friend Journal

https://github.com/user-attachments/assets/137b8a3e-2df1-4c41-8317-ed16139fbd73

# Setup guide
## Create a virtual environment
`conda create --name wisefriend` `conda activate wisefriend`

## Install dependencies
* `conda install pip`
* python dependencies `cd wise-friend` `pip install -r requirements.txt`
* install postgres `conda install -y -c conda-forge postgresql`
* run the postgres docker container `docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16` (ensure to install docker)

## Start the application
* open `start.sh` and fill your API keys
* `sh start.sh`

# Debug
* To connect to the vectore store
`psql -h localhost -p 6024 -U langchain -d langchain` (password: langchain)
