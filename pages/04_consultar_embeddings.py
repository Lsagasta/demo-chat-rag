import streamlit as st
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

# --- CONFIGURACIÓN INICIAL ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Instancio un objeto Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# Sidebar para configuraciones
st.sidebar.header("Configuraciones")
indices = pc.list_indexes()
index_name = st.sidebar.selectbox("Selecciona la base de datos", [indice['name'] for indice in indices])

top_k = st.sidebar.number_input("Número de resultados relevantes (k)", min_value=1, max_value=20, value=5, step=1)

index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Función para obtener el embedding de la pregunta
def obtener_embedding(pregunta):
    return embeddings.embed_query(pregunta)

# Función para obtener resultados de Pinecone
def obtener_resultados(embedding, k):
    response = index.query(vector=embedding, top_k=k, include_metadata=True)
    resultados = [(match['score'], match['metadata']) for match in response['matches']]
    return resultados

# Interfaz principal
st.title("Búsqueda en base de datos vectorial")

# Input para la pregunta
pregunta = st.text_input("Ingresa tu pregunta:")

if pregunta:
    # Generar embedding de la pregunta
    with st.spinner("Generando embedding..."):
        embedding_pregunta = obtener_embedding(pregunta)
    
    st.write("**Embedding de la pregunta:**")
    st.code(embedding_pregunta, language="python")

    # Obtener resultados de la base de datos
    with st.spinner(f"Buscando los {top_k} resultados más relevantes..."):
        resultados = obtener_resultados(embedding_pregunta, top_k)
    
    st.write("**Resultados relevantes:**")
    for score, metadata in resultados:
        st.write(f"- **Score:** {score}")
        st.json(metadata)
