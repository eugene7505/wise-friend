import sys
from datetime import datetime
# import base64

import pandas as pd
import streamlit as st
from langchain import hub

import config
import utils


def display_past_entries(entries):
    st.header("☀️ Your recent mood 🌤️🌦️🌧️⛈️")

    dates = [entry[0].metadata["date"] for entry in entries]
    entries = [entry[0].page_content for entry in entries]
    data = {"Dates": dates, "Entries": entries}
    df = pd.DataFrame(data)
    st.dataframe(df)

def display_reference(top_citations):
    with st.expander(f"**References**"):
        # st.markdown(f"**Reference:**  \n\n*{top_citations}*")
        for i, ref in enumerate(top_citations):
            clean_text = ref.replace('\n', '<br>').replace('\xa0', ' ')
            st.markdown(f"**Reference {i+1}:**", unsafe_allow_html=True)
            st.markdown(clean_text, unsafe_allow_html=True)
            st.markdown("---")

### Streamlit interface
# To start, streamlit run wise_friend_streamlit.py. Add "-- dry-run" to run in dry-run mode.
dry_run = False
arguments = sys.argv[1:]
if arguments:
    dry_run = arguments[0] == "dry-run"
    if dry_run:
        print("Running in dry-run mode")


llm, embeddings = utils.setup_models()
# load wise store and journal store
wise_store, journal_store = utils.load_vector_stores(embeddings)
prompt = hub.pull(config.PROMPT)

st.title("📝 Your Wise Friend Journal")
# Add a new journal entry
st.header("How are you feeling today?")
# User input for journal entry
entry_date = st.date_input("Date", value=datetime.today())
content = st.text_area("Journal entry", "")


if st.button("Reflect"):
    # Store journal entry to the journal_store
    if content:
        utils.add_journal_entry(journal_store, content, str(entry_date))
        st.success("Entry added!")        

        # wise responses
        retrieved_docs = utils.retrieve(content, wise_store) if not dry_run else []
        print(f"Retrieved {len(retrieved_docs)} documents from the wise_repo")
        response = (
            utils.generate(content, retrieved_docs, llm, prompt) if not dry_run else ""
        )
        top_citations = utils.display_top_n_citations(retrieved_docs, response, embeddings, n=2) if not dry_run else ""
        
        st.header(f"Journal Entry {entry_date}")
        with st.chat_message("user", avatar= "✍️"):
            st.markdown(f"**Journal Entry:**  \n\n*{content}*")
        st.header("Wise Friend Response")
        with st.chat_message("ai", avatar= "🧠"):
            st.markdown(f"**Wise Friend:**  \n\n*{response}*")
            display_reference(top_citations)

        # Retrieve relevant journal entries
        entries = (
            utils.get_journal_entries_with_similar(journal_store, content)
            if not dry_run
            else []
        )
        print(f"Retrieved {len(entries)} entries from the journal")
        display_past_entries(entries)
    else:
        st.error("Please enter some content.")

# Sidebar : Upload a document to add to the wise store
uploaded_file = st.sidebar.file_uploader(
    "Add to your wise friends", type=["txt", "pdf"]
)
if uploaded_file is not None:
    file_path = f"./{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    utils.add_wise_entry(wise_store, file_path)
    st.success("Wise friend added!")
