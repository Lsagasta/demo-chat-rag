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
index_name = st.sidebar.selectbox("Selecciona el índice de Pinecone", [indice['name'] for indice in indices])

prompt_system_default = """
Eres un asistente virtual diseñado para ayudar a los usuarios proporcionando respuestas claras y útiles, pero solo debes responder con información obtenida de la base de datos vectorial disponible. Si la información necesaria no está presente en la base de datos o la pregunta está fuera del alcance de los documentos almacenados, informa amablemente que no puedes ayudar con esa solicitud. No inventes ni supongas respuestas fuera de lo que está disponible en la base de datos. Siempre mantén un tono profesional y respetuoso. Si la información no está disponible, ofrece alternativas o sugiere otras fuentes donde el usuario pueda encontrar lo que busca.
"""
prompt_system = st.sidebar.text_area("Define el prompt del sistema", value=prompt_system_default, height=200)

temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.1, 0.1)
max_tokens = st.sidebar.number_input("Máximo de tokens", 50, 2000, 1500, 50)
top_k = st.sidebar.number_input("Número de resultados relevantes (top_k)", 1, 20, 5, 1)

index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def obtener_embedding(pregunta):
    return embeddings.embed_query(pregunta)

def obtener_respuesta_openai(pregunta):
    embedding_pregunta = obtener_embedding(pregunta)
    response = index.query(vector=embedding_pregunta, top_k=top_k, include_metadata=True)

    documentos_relevantes = [match.get('metadata', {}).get('text', '') for match in response['matches']]
    contexto = "\n".join(documentos_relevantes)
    pregunta_y_contexto = f"Pregunta: {pregunta}\n\nContexto:\n{contexto}"

    historial = st.session_state.historial[-20:]
    mensajes = [{"role": "system", "content": prompt_system}] + [
        {"role": msg["role"], "content": msg["content"]} for msg in historial
    ] + [{"role": "user", "content": pregunta_y_contexto}]

    try:
        respuesta_openai = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=mensajes,
            max_tokens=max_tokens,
            temperature=temperature
        )
        respuesta = respuesta_openai.choices[0].message.content.strip()
    except Exception as e:
        respuesta = f"Error al obtener respuesta: {str(e)}"

    st.session_state.historial.append({"role": "user", "content": pregunta})
    st.session_state.historial.append({"role": "assistant", "content": respuesta})

    return respuesta

if "historial" not in st.session_state:
    st.session_state.historial = [{"role": "assistant", "content": "¡Hola! ¿en qué puedo ayudarte hoy?"}]

for message in st.session_state.historial:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pregúntame lo que quieras"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Espera un instante..."):
        respuesta = obtener_respuesta_openai(prompt)
    with st.chat_message("assistant"):
        st.markdown(respuesta)
