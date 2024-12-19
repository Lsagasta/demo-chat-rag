import streamlit as st
from pinecone import Pinecone

# Configuración de Pinecone
pinecone = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# Consultar índices
indices = pinecone.list_indexes()
nombres_indices = [indice['name'] for indice in indices]

st.subheader("Eliminar base de datos")
st.divider()
# Selector de índice
option = st.selectbox("Seleccionar la base de datos a eliminar", nombres_indices)


# Definir función de diálogo con título
@st.dialog("Confirmar eliminación")
def confirm_delete(option):
    st.subheader("Confirmar eliminación")
    st.write(f"¿Está seguro que desea eliminar la base de datos '{option}'?")
    
    if st.button("Confirmar eliminación", type="secondary"):
        try:
            pinecone.delete_index(option)

            # Verificar la eliminación
            indices_actualizados = pinecone.list_indexes()
            nombres_actualizados = [indice['name'] for indice in indices_actualizados]

            if option not in nombres_actualizados:
                st.success(f"La base de datos '{option}' fue eliminada con éxito.")
                st.rerun()
            else:
                st.error(f"No se pudo eliminar la base de datos '{option}'.")
        except Exception as e:
            st.error(f"Error al eliminar la base de datos: {str(e)}")
    
    if st.button("Cancelar"):
        st.info("Operación cancelada.")
        st.rerun()

# Botón para mostrar el cuadro de diálogo
if st.button(f"Eliminar {option}"):
    confirm_delete(option)
