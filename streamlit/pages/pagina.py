import streamlit as st

# Título principal
st.title("📚 Detector de Personas en la Biblioteca - UIDE")

# Imagen de portada
st.image(
    "https://i.pinimg.com/736x/6d/4c/07/6d4c07897314213d0ed46a7c695086f4.jpg",
    caption="Universidad Internacional del Ecuador",
    use_container_width=True
)

# Descripción de la página
st.markdown("""
---
Esta aplicación utiliza **modelos de visión por computadora** para detectar y contar personas en tiempo real  
dentro de la **Biblioteca de la Universidad Internacional del Ecuador**.  

El objetivo principal es ayudar a monitorear el **aforo** y mejorar la **gestión de espacios** en la institución.
---
""")
