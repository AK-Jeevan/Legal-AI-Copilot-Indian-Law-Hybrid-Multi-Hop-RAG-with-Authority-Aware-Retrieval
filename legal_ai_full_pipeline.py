"""
legal_ai_full_pipeline.py

This script builds a complete retrieval‑augmented generation (RAG) pipeline
tailored for Indian legal texts. It handles:
 1. Loading documents from PDFs and CSVs
 2. Creating a hybrid retriever (BM25 + dense embeddings)
 3. Re‑ranking results with a CrossEncoder
 4. Performing multi‑hop retrieval
 5. Generating answers via an LLM chain
 6. Evaluating outputs using faithfulness and relevancy metrics

Every function and statement below is annotated so you can see exactly what
happens at each step, in simple, explanatory language.
"""

import os                     # provides functions for interacting with the operating system
import re                     # regular expressions, used for text splitting and searching
import time                   # unused here but often for benchmarking or timestamps
import pandas as pd           # data analysis library; used for reading CSV files
from typing import List       # typing helpers, not required but good for hints

# pypdf library to read PDF files page-by-page
from pypdf import PdfReader

# LangChain schema objects and retrieval/vectorization helpers
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# OpenAI wrapper for chat-style language models
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Cross-encoder from sentence-transformers for reranking
from sentence_transformers import CrossEncoder

# HuggingFace datasets library for evaluation dataset construction
from datasets import Dataset

# ragas library with evaluation functions and metrics
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy


# ============================================
# 1️⃣ DATA PREPARATION
# ============================================

def load_constitution_pdf(pdf_path):
    """
    Read a PDF of the constitution and split it into separate articles.

    pdf_path: path to a PDF file.

    Returns a list of Document objects, each containing the text
    of one Article and helpful metadata.
    """
    reader = PdfReader(pdf_path)       # open the PDF
    full_text = ""                     # start with empty string

    # iterate through every page and concatenate the extracted text
    for page in reader.pages:
        full_text += page.extract_text() or ""

    # regular expression to identify "Article X" headings
    pattern = r"(Article\s+\d+[A-Z]?\.?)"
    splits = re.split(pattern, full_text)   # split text wherever the pattern appears
    documents = []                          # will hold our Document objects

    # if splitting produced at least one article heading
    if len(splits) > 1:
        # iterate in steps of two: heading then content
        for i in range(1, len(splits), 2):
            article = splits[i]                         # the "Article ..." text
            content = splits[i+1] if i+1 < len(splits) else ""

            # create a LangChain Document with page_content and metadata
            documents.append(
                Document(
                    page_content=article + " " + content,
                    metadata={
                        "source": "Indian Constitution",
                        "type": "constitution",
                        "section": article,
                        "citation": f"Indian Constitution {article}",
                        "authority_score": 3
                    }
                )
            )
    return documents   # return the list of documents extracted


def load_csv_act(csv_path):
    """
    Read a CSV representing a law/act, extract sections and text.

    csv_path: path to a CSV file with at least two columns: section and text.

    Returns a list of Document objects for each non-empty section.
    """
    df = pd.read_csv(csv_path)  # load CSV into a pandas DataFrame
    documents = []              # prepare list for output
    act_name = os.path.basename(csv_path).replace(".csv", "")  # derive act name

    text_col = df.columns[-1]   # assume last column contains the section text
    section_col = df.columns[0] # assume first column contains section numbers

    # iterate through rows of the DataFrame
    for _, row in df.iterrows():
        section = str(row[section_col])  # convert section number to string
        text = str(row[text_col])        # convert text to string

        if not text.strip():             # skip empty text cells
            continue

        # build a Document with content and metadata
        documents.append(
            Document(
                page_content=f"Section {section}. {text}",
                metadata={
                    "source": act_name,
                    "type": "act",
                    "section": f"Section {section}",
                    "citation": f"{act_name} Section {section}",
                    "authority_score": 2
                }
            )
        )
    return documents   # return all the documents read from the CSV


def load_all_documents(folder_path):
    """
    Walk a directory and load all PDFs and CSVs using the helper functions.

    folder_path: path to root folder containing legal documents.

    Returns a flat list of Document objects from every file.
    """
    all_docs = []                             # aggregate documents here
    for file in os.listdir(folder_path):      # loop over each filename
        full_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):             # PDF files → constitution loader
            all_docs.extend(load_constitution_pdf(full_path))
        elif file.endswith(".csv"):            # CSV files → act loader
            all_docs.extend(load_csv_act(full_path))

    return all_docs   # return combined list


# ============================================
# 2️⃣ HYBRID RETRIEVAL
# ============================================

def build_retrievers(documents):
    """
    From a list of Document objects, build two retrieval systems:

      * BM25 (sparse) retriever
      * FAISS vectorstore with HuggingFace embeddings (dense)

    Returns (bm25, dense_retriever).
    """
    bm25 = BM25Retriever.from_documents(documents)  # create BM25 retriever
    bm25.k = 20                                    # set number of docs to return

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
    vectorstore = FAISS.from_documents(documents, embeddings)
    dense = vectorstore.as_retriever(search_kwargs={"k": 20})  # convert to retriever

    return bm25, dense   # return both retrievers


def hybrid_retrieval(query, bm25, dense):
    """
    Retrieve documents using both BM25 and dense retriever, then merge.

    query: user question string
    bm25: sparse retriever returned from build_retrievers
    dense: dense retriever from build_retrievers

    Returns a deduplicated list of Document objects.
    """
    sparse_docs = bm25.get_relevant_documents(query)  # greedy sparse results
    dense_docs = dense.get_relevant_documents(query)  # greedy dense results

    combined = sparse_docs + dense_docs               # concatenate both lists
    unique = {}                                       # temp dict to remove duplicates
    for doc in combined:
        unique[doc.metadata["citation"]] = doc       # key by citation

    return list(unique.values())                      # return unique docs


# ============================================
# 3️⃣ RERANKER
# ============================================

reranker = CrossEncoder("BAAI/bge-reranker-large")  # load a cross‑encoder model once


def rerank_documents(query, documents, top_k=5):
    """
    Use the cross‑encoder to score and sort documents for a given query.

    query: the question string
    documents: list of Document objects to score
    top_k: how many top documents to return (default 5)

    Returns the top_k documents in descending score order.
    """
    pairs = [(query, doc.page_content) for doc in documents]  # make query‑doc pairs
    scores = reranker.predict(pairs)                          # compute relevance scores
    scored = list(zip(documents, scores))                     # pair docs with scores
    scored.sort(key=lambda x: x[1], reverse=True)             # sort by score descending
    return [doc for doc, _ in scored[:top_k]]                # return top_k docs


# ============================================
# 4️⃣ MULTI-HOP
# ============================================

def multi_hop(query, bm25, dense, hops=2):
    """
    Perform a simple multi‑hop retrieval process.

    query: initial question string
    bm25, dense: retrievers from build_retrievers
    hops: number of retrieval iterations to perform (default 2)

    Returns a consolidated list of all documents retrieved across hops.
    """
    context_docs = []            # will accumulate documents across hops
    current_query = query        # start with original query

    for _ in range(hops):        # repeat for desired number of hops
        retrieved = hybrid_retrieval(current_query, bm25, dense)  # get docs
        reranked = rerank_documents(current_query, retrieved)     # rerank them
        context_docs.extend(reranked)                              # keep them

        # build new query by appending first portion of context text
        combined_text = " ".join([doc.page_content for doc in reranked])
        current_query = query + " " + combined_text[:300]

    # deduplicate by citation before returning
    unique = {}
    for doc in context_docs:
        unique[doc.metadata["citation"]] = doc

    return list(unique.values())


# ============================================
# 5️⃣ GENERATION
# ============================================

# instantiate the chat LLM with a fixed temperature for deterministic output
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# define a prompt template that instructs the model to behave as a Legal AI
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a Legal AI specialized in Indian Law.

Use ONLY the provided context.
Cite as [1], [2], etc.

Context:
{context}

Question:
{question}
"""
)

# wrap the LLM and prompt into a reusable chain object
chain = LLMChain(llm=llm, prompt=prompt)


def generate_answer(query, context_docs):
    """
    Produce an answer string by running the LLM chain.

    query: user question
    context_docs: list of Document objects provided as context

    The documents are concatenated with numbered citations before passing
    them to the language model.
    """
    context_text = ""
    for i, doc in enumerate(context_docs):
        context_text += f"[{i+1}] {doc.page_content}\n\n"

    return chain.run({"context": context_text, "question": query})


# ============================================
# 6️⃣ EVALUATION METRICS
# ============================================

def evaluate_rag(query, answer, context_docs, ground_truth):
    """
    Evaluate a single question–answer pair using ragas metrics.

    query: the question string
    answer: model’s generated answer
    context_docs: documents used to generate the answer
    ground_truth: the correct answer text

    Returns a dictionary of evaluation scores.
    """
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [[doc.page_content for doc in context_docs]],
        "ground_truth": [ground_truth]
    }

    dataset = Dataset.from_dict(data)   # create a HF dataset from the dictionary

    result = evaluate(                    # run ragas evaluation
        dataset,
        metrics=[faithfulness, answer_relevancy]
    )

    return result   # return the evaluation results


def recall_at_k(docs, relevant_citations, k=5):
    """
    Compute recall@k: fraction of relevant citations found in the top k documents.

    docs: retrieved Document list
    relevant_citations: list of citation strings that are considered relevant
    k: number of top docs to consider (default 5)
    """
    retrieved = [doc.metadata["citation"] for doc in docs[:k]]
    hits = sum([1 for r in relevant_citations if r in retrieved])
    return hits / len(relevant_citations)


def mean_reciprocal_rank(docs, relevant_citations):
    """
    Compute MRR: reciprocal of the rank of the first relevant document.

    docs: retrieved Document list
    relevant_citations: list of citation strings that are relevant
    """
    for i, doc in enumerate(docs):
        if doc.metadata["citation"] in relevant_citations:
            return 1 / (i + 1)
    return 0


def hallucination_check(answer, context_docs):
    """
    Simple sanity check: ensure the model’s citations reference available docs.

    answer: generated answer text
    context_docs: list of Document objects supplied as context
    """
    citations = re.findall(r'\[(\d+)\]', answer)  # look for numbers in square brackets
    if not citations:
        return "No citations found"
    if max([int(c) for c in citations]) > len(context_docs):
        return "Possible Hallucination"
    return "No Hallucination"


# ============================================
# 7️⃣ ENTRYPOINT / PIPELINE RUNNER
# ============================================

if __name__ == "__main__":
    # folder containing PDFs and CSVs to index; adjust as needed
    docs_folder = "path/to/legal_documents"

    # load all legal documents into memory
    documents = load_all_documents(docs_folder)
    print(f"Loaded {len(documents)} documents from {docs_folder}")

    # build both sparse and dense retrievers once
    bm25, dense = build_retrievers(documents)

    # prompt the user for a question; strip whitespace
    user_query = input("Enter your legal question: ").strip()
    if not user_query:
        print("No query provided, exiting.")
    else:
        # run a simple two-hop retrieval to gather relevant context
        context_docs = multi_hop(user_query, bm25, dense, hops=2)
        print("Retrieved the following context citations:")
        for idx, doc in enumerate(context_docs, start=1):
            print(f" [{idx}] {doc.metadata['citation']}")

        # generate an answer using the LLM chain
        answer = generate_answer(user_query, context_docs)
        print("\n=== Model Answer ===\n", answer)

        # optional: ask for ground truth to compute evaluation metrics
        gt = input("\nProvide ground truth answer for evaluation (or press enter to skip): ").strip()
        if gt:
            metrics = evaluate_rag(user_query, answer, context_docs, gt)
            print("\nEvaluation metrics:", metrics)

        # perform a simple hallucination sanity check
        print("Hallucination check:", hallucination_check(answer, context_docs))