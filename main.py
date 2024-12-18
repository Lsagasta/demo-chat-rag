import streamlit as st
from pinecone import Pinecone

# Configuración de Pinecone
pinecone = Pinecone(api_key="pcsk_5xk6d7_NLDEknysnpANqSxmRdzzNpgiPZn6czJwje4Jw9dNwMaNnxe7iAKnfYmA6AciDnh")
INDEX_NAME = 'pruebas'
nombres_indices = ""

st.header("Configuración de Database", divider = "gray")

st.subheader("Consultar existencia de base de datos")
st.write("En esta versión de prueba sólo se permite un máximo de 5 base de datos")


indices = pinecone.list_indexes()
nombres_indices = [indice['name'] for indice in indices]
cantidad = len(nombres_indices)
st.write(f"{cantidad} base de datos existentes: ")
st.code(nombres_indices)

st.divider()









