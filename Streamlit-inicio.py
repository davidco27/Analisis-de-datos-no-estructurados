#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import streamlit as st
import pandas as pd
import webbrowser

opciones = [
    ("Inicio", ""),  # Opción predeterminada
    ("OpenAI", "https://www.openai.com"),
    ("GitHub", "https://github.com"),
    ("Streamlit", "https://streamlit.io")
]
opcion = st.sidebar.selectbox("Elige un sitio web para visitar:", opciones, format_func=lambda x: x[0])
if opcion[1]:
    webbrowser.open(opcion[1])
_,col_img,_,_,_ = st.columns(5)
with col_img:
    st.image("imagenesDesign/logo-icai.png", width = 300)
st.title("Práctica final No estructurados")
st.subheader("Javier Álvarez Martínez y David Cocero Quintanilla")
st.divider()
st.write("Presentamos aquí nuestra interfaz para permitir a los usuarios explorar nuestro trabajo final de la asignatura que se divide en 2 componentes principales: Texto e Imagen. Para empezar seleccione una de las 2 opciones que aparecen en el desplegable de la izquierda.")
st.header("ÍNDICE")
st.write("Mostramos aquí las implementaciones para los dos tipos de datos")
col1_ini, col2_ini = st.columns(2)
with(col1_ini):
    st.subheader("TEXTO")
    st.markdown("""
* EDA (Análisis exploratorio del dataset)
* Generador de resúmenes:
* Encoder - Decoder from scratch
* Modelo summarization Hugging Face
* Few Shot Modelo generativo Hugging Face
* Clasificador de resúmenes por temas
* Buscador de noticias por palabras clave
""")
with (col2_ini):
    st.subheader("IMAGEN")
    st.markdown("""
* Clasificador de imágenes deportivas:
* Red convolucional from scratch
* Modelo de transfer learning
* Generador de imágenes (GAN)
""")
