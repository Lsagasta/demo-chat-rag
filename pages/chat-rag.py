import streamlit as st
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

# --- CONFIGURACIÓN INICIAL ---
# Obtengo las claves API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Instancio un objeto Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# Sidebar para configuraciones
st.sidebar.header("Configuraciones")

# Configuración del índice de Pinecone
indices = pc.list_indexes()
nombres_indices = [indice['name'] for indice in indices]
index_name = st.sidebar.selectbox("Selecciona el índice de Pinecone", nombres_indices)

# Configuración del prompt del sistema
prompt_system_default = """
Eres un asistente virtual diseñado para ayudar a los usuarios proporcionando respuestas claras y útiles. Si un usuario hace una pregunta fuera del alcance de tu conocimiento o de tus capacidades, informa amablemente que no puedes ayudar con esa solicitud. Limítate a responder con información relevante para el contexto de la conversación y mantén un tono profesional y respetuoso en todo momento. Si es necesario, ofrece alternativas para que el usuario pueda obtener la información que necesita.
"""
prompt_system = st.sidebar.text_area("Define el prompt del sistema", value=prompt_system_default, height=200)

# Configuración de parámetros del modelo de OpenAI
st.sidebar.subheader("Configuración del modelo OpenAI")
temperature = st.sidebar.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
max_tokens = st.sidebar.number_input("Máximo de tokens", min_value=50, max_value=2000, value=1500, step=50)
top_k = st.sidebar.number_input("Número de resultados relevantes (top_k)", min_value=1, max_value=20, value=5, step=1)

# Conectar al índice seleccionado
index = pc.Index(index_name)

# Defino el modelo de embeddings de OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# --- FUNCIONES ---
def obtener_embedding(pregunta):
    embedding = embeddings.embed_query(pregunta)
    return embedding

def obtener_respuesta_openai(pregunta: str):
    # Obtener embedding de la pregunta
    embedding_pregunta = obtener_embedding(pregunta)

    # Buscar los vectores más similares en Pinecone
    response = index.query(vector=embedding_pregunta, top_k=top_k, include_metadata=True)

    # Extraer documentos más relevantes
    documentos_relevantes = [match.get('metadata', {}).get('text', '') for match in response['matches']]

    # Crear contexto a partir de los documentos relevantes
    contexto = "\n".join(documentos_relevantes)
    pregunta_and_contexto = f"Pregunta: {pregunta}\n\nContexto:\n{contexto}"

    # Mantén solo los últimos N mensajes en el historial
    N = 20
    st.session_state.historial = st.session_state.historial[-N:]

    # Formatear el historial como texto
    historial_texto = "\n".join([f"{mensaje['role']}: {mensaje['content']}" for mensaje in st.session_state.historial])

    # Obtener respuesta de OpenAI utilizando el método ChatCompletion.create()
    respuesta_openai = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Cambia el modelo si es necesario
        messages=[
            {"role": "system", "content": f"{prompt_system}\n\nHistorial:\n{historial_texto}"}
        ] + [{"role": "user", "content": pregunta_and_contexto}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    respuesta = respuesta_openai.choices[0].message["content"].strip()
    respuesta = respuesta.replace("$", "\\$")

    # Actualizar historial
    st.session_state.historial.append({"role": "user", "content": pregunta})
    st.session_state.historial.append({"role": "assistant", "content": respuesta})

    return respuesta

# --- INTERFAZ DE USUARIO ---
# Inicializar historial
if "historial" not in st.session_state:
    st.session_state.historial = [
        {"role": "assistant", "content": "¡Hola! ¿en qué puedo ayudarte hoy?"}
    ]

# Mostrar historial de chat
for message in st.session_state.historial:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input("Pregúntame lo que quieras"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Espera un instante..."):
        respuesta = obtener_respuesta_openai(prompt)
    with st.chat_message("assistant"):
        st.markdown(respuesta)
