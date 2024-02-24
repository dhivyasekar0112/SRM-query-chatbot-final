pip install -i https://pypi.org/simple/ bitsandbytes
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import os
import transformers
from langchain.vectorstores import Qdrant
from langchain.chains import VectorDBQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
from torch import cuda, bfloat16
import torch
import locale

# Install necessary dependencies
st.title("PDF QA Bot with Streamlit")

# Define functions and classes
def load_llama_pipeline():
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    hf_auth = "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"
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
                            model_id,
                            use_auth_token=hf_auth)
    model.eval()
    query_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",)
    return HuggingFacePipeline(pipeline=query_pipeline)

def load_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

def load_documents(root):
    text = ""
    for f in os.listdir(root):
        pdf_path = os.path.join(root, f)
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def initialize_qa(llm, docs):
    embeddings = load_embeddings()
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1024,
        chunk_overlap=20,
        length_function=len,)
    docs = text_splitter.split_text(docs)
    doc_store = Qdrant.from_texts(
        docs,
        embeddings,
        path="/vectors1",
        collection_name="my_documents",
    )
    qa = VectorDBQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        vectorstore=doc_store,
    )
    return qa

# Load models and data
llm = load_llama_pipeline()
root = '/content/PDFs_1'
docs = load_documents(root)
qa = initialize_qa(llm, docs)

# Define Streamlit UI
st.sidebar.title("PDF QA Bot")
query = st.sidebar.text_input("Enter your question here:")

# Execute query and display results
if st.sidebar.button("Ask"):
    st.write(f"Query: {query}")
    result = qa.run(query)
    st.write("Result:", result)
