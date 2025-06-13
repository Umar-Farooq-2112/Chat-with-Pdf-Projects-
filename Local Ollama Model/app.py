import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import os

# from dotenv import load_dotenv
# load_dotenv()

# os.environ['GOOGLE_API_KEY'] = os.getenv("API_KEY")

def get_text_from_pdf(all_pdf):
    text = ''
    for pdf in all_pdf:
        pdf_rdr = PdfReader(pdf)
        for page in pdf_rdr.pages:
            text += page.extract_text()
    return text

def get_text_chunks_from_text(input_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks = text_splitter.split_text(input_text)
    return chunks

def get_vector_store_from_chunks(text_chunks):
    embeddings = OllamaEmbeddings(model='deepseek-r1:1.5b')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_temp = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = Ollama(model='deepseek-r1:1.5b', temperature=0.3)
    prompt = PromptTemplate(template=prompt_temp, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def user_input_handling(question):
    embeddings = OllamaEmbeddings(model='deepseek-r1:1.5b')
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("No FAISS index found. Please upload and submit your PDF files first.")
        return
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain(
        {
            "input_documents": docs,
            "question": question
        },
        return_only_outputs=True
    )
    print(response)
    st.write("Reply:  ", response['output_text'])

def main():
    st.set_page_config("Chat PDF using Google Gemini")

    question = st.text_input("Ask your Question from PDF Files")

    if question:
        user_input_handling(question)

    with st.sidebar:
        st.title("PDF Menu")

        pdf_docs = st.file_uploader("Upload your PDF Files and click Submit", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Processing: Reading PDF..."):
                st.info("Extracting text from PDF...")
                raw = get_text_from_pdf(pdf_docs)
                st.info("Splitting text into chunks...")
                text_chunks = get_text_chunks_from_text(raw)
                st.info("Generating embeddings and saving vector store...")
                get_vector_store_from_chunks(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
