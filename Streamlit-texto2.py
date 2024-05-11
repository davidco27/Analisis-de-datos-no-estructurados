#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline
from funciones_streamlit import generar_resumen
import webbrowser

@st.cache_resource
def load_resources():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_resources()
st.sidebar.link_button(label="Inicio", url="https://no-estructurados-inicio.streamlit.app")
st.sidebar.link_button(label="Imagen", url="https://no-estructurados-imagen.streamlit.app")
st.sidebar.link_button(label="Texto", url="https://no-estructurados-texto1.streamlit.app")
st.header("Resumen de las noticias")
st.write("Hemos implementado diferentes modelos para crear resúmenes de noticias. Para probarlo, empieza elegiendo uno de los tres modelos: con arquitectura Encoder-Decoder from Scratch, Modelo de Hugging Face, Few-shot con Hugging Face.")
st.write("Para medir la eficiencia de los resumenes hemos utilizado BLEU. BLEU es una métrica de evaluación de la calidad de traducción automática que compara un texto generado con uno de referencia, calculando la precisión de las n-gramas coincidentes. Cuanto más alto es el puntaje BLEU, más similar es el texto generado al texto de referencia.")
modelos = ["Encoder-Decoder","Hugging Face","Few-shot"]
modelo_seleccionado = st.radio("Selecciona un modelo:", modelos,index=modelos.index("Hugging Face"))
if modelo_seleccionado == "Encoder-Decoder":
    webbrowser.open("https://no-estructurados-texto1.streamlit.app/?valor=Encoder-Decoder")
if modelo_seleccionado == "Few-shot":
    webbrowser.open("https://no-estructurados-texto1.streamlit.app/?valor=Few-shot")
txt = st.text_area(
"Pega aquí la noticia a resumir (tiene que ser en inglés)",height=500
)
st.write("El resumen no puede ser más largo de 200 palabras ")
rsm = st.text_area("Introduce el resumen ")
st.write(f"La longitud de la noticia es de {len(txt.split())} palabras y la de tu resumen de {len(rsm.split())} " )
categorias = ["business","entertainment","politics","sport","tech"]
cat = st.selectbox("Selecciona la categoria de la noticia", categorias)
st.write("En este modelo de texto, se ha utilizado un modelo preentrenado de HuggingFace preparado para resumir. Para seleccionar la longitud de los resúmenes lo que hemos hecho es utilizar como máximo las longitudes medias de resúmenes que teníamos en los datos de partida y como valor mínimo de longitud hemos usado el máximo menos 30 aproximadamente.")
st.write("""El modelo "facebook/bart-large-cnn" es parte de la familia BART (BART: Bidirectional and Auto-Regressive Transformers), desarrollado por Facebook AI. BART es un modelo basado en la arquitectura Transformer que ha demostrado ser efectivo en tareas de generación de lenguaje, traducción, resumen de texto...""")
st.write("""BART es un modelo transformer encoder-encoder (seq2seq) con un encoder bidireccional (similar a BERT) y un decoder autoregresivo (similar a GPT). BART se pre-entrena mediante la corrupción de texto con una función de ruido arbitraria y el aprendizaje de un modelo para reconstruir el texto original. """)
btn_gen = st.button("Generar Resumen con modelo de Hugging Face")
if btn_gen:
    if len(txt) == 0  or len(rsm) == 0:
        st.warning("¡Cuidado! Introduzca una noticia y un resumen para continuar.")
    else:
        with st.spinner(text="Generando el resumen..."):
            rsm_gen,score = generar_resumen(summarizer,txt,cat,rsm)
            st.subheader("Resultados")
            st.markdown("***Resumen Generado***")
            st.write(rsm_gen)
            st.markdown("***BLEU score obtenido***")
            st.write(f"{score*100:.2f} %")
        