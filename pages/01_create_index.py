import streamlit as st
from pinecone import Pinecone
import re

# Configuración de Pinecone


# Instancio un objeto Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
st.header("Crear un índice en Pinecone")

# Entrada del nombre del índice
index_name = st.text_input("Nombre del índice (letras minúsculas, números y '-')")

# Validación de caracteres permitidos
def validar_nombre(nombre):
    return bool(re.match(r'^[a-z0-9\-]+$', nombre))

# Configuración del índice
dimension = st.number_input("Dimensión del índice (text-embedding-ada-002 : 1536 dimensiones)", min_value=1, step=1, value=1536)
st.write("")
metric = st.selectbox("Métrica de similitud", ["cosine", "euclidean", "dotproduct"])
cloud = st.selectbox("Proveedor de nube", ["aws", "gcp", "azure"])
region = st.selectbox("Región", ["us-east-1", "us-west-1", "europe-west1"])

if st.button("Crear índice"):
    if not validar_nombre(index_name):
        st.error("El nombre del índice solo puede contener letras minúsculas, números y guiones ('-').")
    else:
        try:
            # Crear índice
            pinecone.create_index(
                name=index_name, 
                dimension=dimension, 
                metric=metric,
                spec={"serverless": {"cloud": cloud, "region": region}}
            )

            # Verificar si se creó correctamente
            indices_actualizados = pinecone.list_indexes()
            nombres_actualizados = [indice['name'] for indice in indices_actualizados]

            if index_name in nombres_actualizados:
                st.success(f"Índice '{index_name}' creado con éxito.")
            else:
                st.error(f"No se pudo crear el índice '{index_name}'.")
        except Exception as e:
            st.error(f"Error al crear el índice: {str(e)}")
