import streamlit as st
from datetime import datetime
import utils as wb
import config

from langchain import hub
import pandas as pd

def display_entries(entries):
    st.header("Journal Entries Similar to Your Current Mood")

    dates = [entry[0].metadata['date'] for entry in entries]
    entries = [entry[0].page_content for entry in entries]
    data = {'Dates': dates, 'Entries': entries}
    df = pd.DataFrame(data)
    st.dataframe(df) 


### Streamlit interface
st.title("Your Wise Friend Journal")

llm, embeddings = wb.setup_models()
# init wise_repo and journal store
wise_store, journal_store = wb.setup_vector_stores(embeddings)
prompt = hub.pull(config.PROMPT)

# Add a new journal entry
st.header("Add a New Journal Entry")
entry_date = st.date_input("Date", value=datetime.today())
content = st.text_area("Entry", "")

if st.button("Add Entry"):
    if content:
        wb.add_journal_entry(journal_store, content, str(entry_date))
        st.success("Entry added!")

        # journal entries
        entries = wb.get_journal_entries_with_similar(journal_store, content)
        st.info(f"Retrieved {len(entries)} entries from the journal")
        display_entries(entries)

        # wise responses
        retrieved_docs = wb.retrieve(content, wise_store)
        st.info(f"Retrieved {len(retrieved_docs)} documents from the wise_repo")
        response = wb.generate(content, retrieved_docs, llm, prompt)
        st.write(f"Your wise friend says: {response}")
    else:
        st.error("Please enter some content.")
