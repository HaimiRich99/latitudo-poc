import streamlit as st
from llm_backend import parse_query_with_rag
#from llm_backend_groq import parse_query_with_rag
from stac_api_backend import *
from utils import *

print('Running')  # This will run on every Streamlit rerun — expected

st.set_page_config(page_title="🛰️ Satellite Assistant", layout="centered")
st.title("🛰️ Satellite Data Assistant")

# Initialize session state
if "parsed_query" not in st.session_state:
    st.session_state.parsed_query = None
if "error" not in st.session_state:
    st.session_state.error = None

# User input
query = st.text_input("Ask your satellite question:", placeholder="E.g., NDVI over Naples in July 2023")

# Only process if the input is different from the last one
if query and (st.session_state.parsed_query is None or st.session_state.last_query != query):
    with st.spinner("Thinking..."):
        parsed, text, error = parse_query_with_rag(query)
        st.session_state.last_query = query  # Save to avoid reprocessing same input

        if parsed:
            st.session_state.parsed_query = parsed
            st.session_state.text = text
            st.session_state.error = None
        else:
            st.session_state.error = error
            st.session_state.parsed_query = None
            st.session_state.text = None

# Show error or extracted results
if st.session_state.error:
    st.warning(st.session_state.error)

if st.session_state.parsed_query:
    st.success("✅ Extracted Parameters")
    #st.json(st.session_state.parsed_query)
    if st.session_state.text:
        st.markdown(st.session_state.text)

    if st.button("Run Satellite Query"):
        with st.spinner("Running satellite backend..."):
            results_list = stac_api_query(
                location=st.session_state.parsed_query['location'],
                start_date=st.session_state.parsed_query['start_date'],
                end_date=st.session_state.parsed_query.get('end_date', None),
                cloud_cover=st.session_state.parsed_query.get('cloud_cover', 10)
            )

            time_retrieval_type = st.session_state.parsed_query.get('time_retrieval_type', 'single')
            bands=st.session_state.parsed_query['frequencies']
            frequency_operation = st.session_state.parsed_query.get('frequency_operation', None)
            if time_retrieval_type == 'single':
                mappa = results_list[0]
                fig = decider(mappa, bands, frequency_operation)
                st.pyplot(fig)
            elif time_retrieval_type == 'compare':
                mappa1 = results_list[0]
                mappa2 = results_list[-1]
                fig = decider(mappa1, bands, frequency_operation)
                fig2 = decider(mappa2, bands, frequency_operation)
                st.pyplot(fig)
                st.pyplot(fig2)
            elif time_retrieval_type == 'time-lapse':
                st.success("Time-lapse mode is not yet implemented.")
                #for mappa in results_list:
                #    fig = decider(mappa, bands, frequency_operation)
                #    st.pyplot(fig)

            

            
                    
                    
