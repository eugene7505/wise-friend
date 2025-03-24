import sys
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain import hub

import config
import utils


def display_entries(entries):
    st.header("Your Recent Mood")

    dates = [entry[0].metadata["date"] for entry in entries]
    entries = [entry[0].page_content for entry in entries]
    data = {"Dates": dates, "Entries": entries}
    df = pd.DataFrame(data)
    st.dataframe(df)


### Streamlit interface
# To start, streamlit run wise_friend_streamlit.py. Add "-- dry-run" to run in dry-run mode.
dry_run = False
arguments = sys.argv[1:]
if arguments:
    dry_run = arguments[0] == "dry-run"
    if dry_run:
        print("Running in dry-run mode")

st.title("Your Wise Friend Journal")

llm, embeddings = utils.setup_models()
# load wise store and journal store
wise_store, journal_store = utils.load_vector_stores(embeddings)
prompt = hub.pull(config.PROMPT)

# Add a new journal entry
st.header("Add a New Journal Entry")
entry_date = st.date_input("Date", value=datetime.today())
content = st.text_area("Entry", "")

if st.button("Add Entry"):
    if content:
        utils.add_journal_entry(journal_store, content, str(entry_date))
        st.success("Entry added!")

        # journal entries
        entries = (
            utils.get_journal_entries_with_similar(journal_store, content)
            if not dry_run
            else []
        )
        print(f"Retrieved {len(entries)} entries from the journal")
        display_entries(entries)

        # wise responses
        retrieved_docs = utils.retrieve(content, wise_store) if not dry_run else []
        print(f"Retrieved {len(retrieved_docs)} documents from the wise_repo")
        response = (
            utils.generate(content, retrieved_docs, llm, prompt) if not dry_run else ""
        )
        st.write(f"Your wise friend says: {response}")
        st.write(f"Reference: {utils.display_top_n_citations(retrieved_docs, response, embeddings, n=2)}")
    else:
        st.error("Please enter some content.")

uploaded_file = st.sidebar.file_uploader(
    "Add to your wise friends", type=["txt", "pdf"]
)
if uploaded_file is not None:
    file_path = f"./{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    utils.add_wise_entry(wise_store, file_path)
    st.success("Wise friend added!")
