"""
evaluation_dashboard.py

A simple Streamlit web interface that lets you run the
Legal‑AI pipeline interactively and view basic evaluation
metrics.

This dashboard:
  * Accepts a legal query from the user
  * Loads documents and builds retrievers on demand
  * Performs hybrid + multi‑hop retrieval
  * Generates an answer with the LLM chain
  * Measures latency and runs ragas metrics
  * Displays results, metrics and a hallucination check

Copy or adapt this file when you want a quick GUI for manual
experiments and demonstrations.
"""

import streamlit as st          # streamlit for building the web UI
import pandas as pd             # pandas for tabular display of metrics
import time                     # used to measure latency
from legal_ai_full_pipeline import *   # import all functions from the pipeline

# Configure the Streamlit page title appearing in the browser tab
st.set_page_config(page_title="Legal AI Evaluation Dashboard")
# Show a main title at the top of the page
st.title("📊 Legal AI Copilot - Evaluation Dashboard")

# text input widget where the user types their legal question
query = st.text_input("Enter Legal Query")
# path to folder containing legal documents to index
folder_path = "/mnt/data"   # adjust this to your actual documents location

# when the user clicks the button, execute the evaluation logic
if st.button("Run Evaluation"):

    start_time = time.time()   # record start time for latency measurement

    # load documents and set up retrievers every time the button is pressed
    documents = load_all_documents(folder_path)
    bm25, dense = build_retrievers(documents)

    # perform two‑hop hybrid retrieval to get context documents
    context_docs = multi_hop(query, bm25, dense)
    # generate an answer using the LLM chain
    answer = generate_answer(query, context_docs)

    latency = time.time() - start_time   # compute elapsed time in seconds

    # Example ground truth answer; replace with real data for proper evaluation
    ground_truth = "Murder is defined under IPC Section 300"

    # compute ragas evaluation scores using the temporary ground truth
    rag_scores = evaluate_rag(query, answer, context_docs, ground_truth)

    # perform the hallucination check on the generated answer
    hallucination_status = hallucination_check(answer, context_docs)

    # display the generated answer in the dashboard
    st.subheader("🧠 Generated Answer")
    st.write(answer)

    # prepare a small table of metrics to show
    metrics_data = {
        "Metric": [
            "Latency (sec)",
            "Faithfulness",
            "Answer Relevancy"
        ],
        "Value": [
            round(latency, 2),
            float(rag_scores["faithfulness"][0]),
            float(rag_scores["answer_relevancy"][0])
        ]
    }

    # render the metrics table
    st.subheader("📈 Metrics")
    st.table(pd.DataFrame(metrics_data))

    # show the hallucination check result
    st.subheader("🚨 Hallucination Check")
    st.write(hallucination_status)

    # list each retrieved citation with its index
    st.subheader("📚 Retrieved Citations")
    for i, doc in enumerate(context_docs):
        st.write(f"[{i+1}] {doc.metadata['citation']}")