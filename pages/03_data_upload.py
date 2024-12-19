import streamlit as st
from pinecone import Pinecone
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
import os


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def crear_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def crear_embeddings(chunks, option):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("La clave de API de OpenAI no está configurada.")
        st.stop()
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key, 
        model="text-embedding-ada-002"
    )
    documents = [Document(page_content=chunk) for chunk in chunks]
    PineconeVectorStore.from_documents(
        documents,
        embedding=embeddings,
        index_name=option
    )

# Inicializar 
pinecone = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])


st.subheader("1. Seleccionar documentos")
pdf_docs = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
raw_text = get_pdf_text(pdf_docs)

with st.expander("Ver Texto Extraido"):
    st.write(raw_text)

st.divider()


st.subheader("2. Seleccionar base a la cual se subirá el documento")

# Consultar índices
indices = pinecone.list_indexes()
nombres_indices = [indice['name'] for indice in indices]

# Selector de índice
def actualizar_option():
    st.session_state.option = st.session_state.seleccion

st.selectbox(
    "Bases:", nombres_indices, key="seleccion", on_change=actualizar_option)

st.divider()

# Inicializar estado para chunks y opción
if "chunks" not in st.session_state:
    st.session_state.chunks = ""
if "option" not in st.session_state:
    st.session_state.option = None

st.subheader("3. Crear chunks")
if st.button("Crear Chunks"):
    with st.expander("Ver Chunk"):
        st.session_state.chunks = crear_chunks(raw_text)
        st.write(st.session_state.chunks)

st.divider()

st.subheader("4. Crear Embeddings")
if st.button("Crear Embeddings"):
    if not st.session_state.chunks:
        st.error("Primero debes crear los chunks.")
    elif not st.session_state.option:
        st.error("Selecciona una base antes de continuar.")
    else:
        crear_embeddings(st.session_state.chunks, st.session_state.option)
        st.success("Embeddings creados correctamente.")
