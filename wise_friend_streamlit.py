import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st
import utils
from sqlalchemy import create_engine
from streamlit_google_auth import Authenticate
import json
import config

from langsmith import Client
import logging
import asyncio

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, ERROR
    format="%(asctime)s [%(levelname)s] %(funcName)s - Line %(lineno)d: %(message)s",
    handlers=[
        logging.FileHandler("wise_friend_streamlit.log"),
        logging.StreamHandler(),  # Also print to console
    ],
)

logger = logging.getLogger(__name__)

# Initialize the Langsmith client for logging user feedback
client = Client()
db_engine = create_engine(config.PSQL_URL)

# Initialize Streamlit session state to manage the app's state across runs
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "journal_entry" not in st.session_state:
    st.session_state.journal_entry = ""
if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""
if "stage" not in st.session_state:
    st.session_state.stage = 0
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "wise_collection" not in st.session_state:
    st.session_state.wise_collection = []
if "dry_run" not in st.session_state:
    st.session_state.dry_run = False
if "top_citations" not in st.session_state:
    st.session_state.top_citations = ""


# Function to set the state of app to control the interface flow.
# The app has three stages:
# 0: waiting for entry. We display previous responses if exist. Upon the "Reflect" button is clicked, we transition to stage 1.
# 1: Generate & display wise response by making LLM call. Once added, then transition back to stage 0
def set_state(i):
    st.session_state.stage = i


# Function to display feedback button under the wise response
def display_feedback_button():
    st.feedback(options="thumbs", key="user_feedback", on_change=log_user_feedback)


# Function to log user feedback to Langsmith
# This function is called when the user clicks the feedback button
# It logs the feedback score (0: thumbs down, 1: thumbs up) and the comment (if any) to Langsmith
def log_user_feedback():
    if not st.session_state.dry_run:
        client.create_feedback(
            st.session_state.response_run_id,
            key="feedback-key",
            score=st.session_state.user_feedback,
            # comment="comment", # TODO: add open text box for user comment
        )
    else:
        st.toast("Feedback logged in dry-run mode.")
    set_state(0)


# Function to display necessary information on the frondend including
# the journal entry, wise response, reference, past entries and wise collection.
def display_journal_entry(entry, date):
    st.header(f"Journal Entry {date}")
    with st.chat_message("user", avatar="✍️"):
        st.markdown(f"**Journal Entry:**  \n\n*{entry}*")


async def stream_response(query, retrieved_docs, llm, prompt, response_placeholder):
    response_text = ""
    async for token, run_id in utils.generate_streaming(
        query, retrieved_docs, llm, prompt
    ):
        response_text += token
        response_placeholder.write(response_text)
        if run_id:
            st.session_state.response_run_id = run_id
    return response_text


def display_wise_response():
    st.header("Wise Friend Response")
    with st.chat_message("ai", avatar="🧠"):
        st.markdown("**Wise Friend:**  \n\n")

        if st.session_state.llm_response:
            st.write(st.session_state.llm_response)
        else:
            response_placeholder = st.empty()
            response_text = asyncio.run(
                stream_response(
                    st.session_state.journal_entry,
                    retrieved_docs,
                    llm,
                    utils.prompt,
                    response_placeholder,
                )
            )
            st.session_state.llm_response = response_text
            display_feedback_button()


def display_reference(top_citations):
    with st.expander("**References**"):
        for i, ref in enumerate(top_citations):
            clean_text = ref.replace("\n", "<br>").replace("\xa0", " ")
            st.markdown(f"**Reference {i + 1}:**", unsafe_allow_html=True)
            st.markdown(clean_text, unsafe_allow_html=True)
            st.markdown("---")


def display_entries(entries):
    df = pd.DataFrame(entries, columns=["Entry", "Date"])
    st.dataframe(df, hide_index=True)


# TODO: Setup secret management https://docs.streamlit.io/develop/concepts/connections/secrets-management
with open("client_secret.json", "w") as f:
    json.dump(json.loads(st.secrets["google"]["client_secret_json"]), f)
authenticator = Authenticate(
    secret_credentials_path="client_secret.json",
    cookie_name="my_cookie_name",  # TODO: do we need this?
    cookie_key="this_is_secret",  # TODO: do we need this?
    redirect_uri="http://localhost:8501",
)

### Streamlit interface
# startup
llm, embeddings = utils.setup_models()
# load the stores for the logged-in user. For logged-out users, use the default test account
userid = (
    st.session_state.user_info.get("id")
    if st.session_state.connected
    else config.TEST_USER_ID
)
wise_store, journal_store = utils.load_vector_stores(embeddings, userid)
collection_table = utils.get_collection_table(db_engine)

# To start, streamlit run wise_friend_streamlit.py. Add "-- dry-run" to run in dry-run mode.
if not st.session_state.initialized:
    arguments = sys.argv[1:]
    if arguments:
        st.session_state.dry_run = arguments[0] == "dry-run"
    logger.info(f"Running in dry-run mode? {st.session_state.dry_run}")
st.session_state.initialized = True

# Default view for the app for users to enter their journal entry
st.title("📝 Your Wise Friend Journal")
# Add a new journal entry
st.header("How are you feeling today?")
# User input for journal entry
st.session_state.date = str(st.date_input("Date", value=datetime.today()))
st.session_state.journal_entry = st.text_area(
    "Journal entry", st.session_state.journal_entry
)
st.button("Reflect", on_click=set_state, args=[1])

if st.session_state.llm_response:
    display_wise_response()
    display_reference(st.session_state.top_citations)
    if st.session_state.entries:
        st.header("☀️ Your recent mood 🌤️🌦️🌧️⛈️")
        # Display the entries
        display_entries(st.session_state.entries)

# Store the journal entry and generate wise response once the user clicks the "Reflect" button
if st.session_state.stage == 1:
    # Store journal entry to the journal_store
    if st.session_state.journal_entry:
        utils.add_journal_entry(
            journal_store,
            st.session_state.journal_entry,
            st.session_state.date,
            st.session_state.dry_run,
        )
        st.success("Entry added!")

        # wise responses
        retrieved_docs = utils.retrieve(
            st.session_state.journal_entry, wise_store, st.session_state.dry_run
        )
        logger.info(f"Retrieved {len(retrieved_docs)} documents from the wise_repo")

        if not st.session_state.dry_run:
            if len(retrieved_docs) == 0:
                st.session_state.llm_response = (
                    "Hello, you haven't added any wise friends. Do you need some help?"
                )
                # display static llm response
                display_wise_response()
            else:
                # display streaming llm response
                display_wise_response()
                st.session_state.top_citations = utils.display_top_n_citations(
                    retrieved_docs, st.session_state.llm_response, embeddings, n=2
                )
                display_reference(st.session_state.top_citations)

        # Retrieve relevant journal entries
        st.session_state.entries = utils.get_journal_entries(db_engine, userid)
        logger.info(
            f"Retrieved {len(st.session_state.entries)} entries from the journal"
        )
        if st.session_state.entries:
            st.header("☀️ Your recent mood 🌤️🌦️🌧️⛈️")
            # Display the entries
            display_entries(st.session_state.entries)
    else:
        st.error("Please enter some content.")


# Sidebar: Upload a document to add to the wise store
with st.sidebar:
    uploaded_file = st.file_uploader("Add to your wise friends", type=["txt", "pdf"])
    with st.expander("📚 Your collections"):
        display_entries(st.session_state.wise_collection)
    if uploaded_file is not None:
        # ensure it's a new file and we want to add to the collection
        if st.session_state.uploaded_file is None or (
            uploaded_file.name != st.session_state.uploaded_file.name
        ):
            logger.info(f"Adding {uploaded_file} to the collection")
            file_path = f"/tmp/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            utils.add_wise_entry(wise_store, file_path)
            utils.update_collection(
                collection_table,
                userid,
                file_path.split("/")[-1],
                st.session_state.date,
                db_engine,
            )
            st.success("Wise friend added!")
            # update states
            st.session_state.wise_collection += [
                (uploaded_file.name, st.session_state.date)
            ]
            logger.info(
                f"Updating states: wise_collection = {st.session_state.wise_collection}, uploaded_file = {st.session_state.uploaded_file}"
            )
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")

    # Catch the login event
    authenticator.check_authentification()
    # Create the login button
    authenticator.login()

    if st.session_state.connected:
        st.write(f"Hello, {st.session_state.user_info.get('name')} ☀️")
        if st.button("Log out"):
            authenticator.logout()
