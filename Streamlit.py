#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import streamlit as st
import pandas as pd
import ast
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from transformers import pipeline
from funciones_streamlit import generar_resumen,generar_resumen_few_shot,generar_resumen_lstm
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

@st.cache_resource
def load_resources():
    model_name='google/flan-t5-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model,tokenizer,summarizer

model, tokenizer,summarizer = load_resources()
folders_in_directory = ["Texto","Imagen"]
folders_in_directory.insert(0, "Sin selección")
selected_folder = st.sidebar.selectbox("Selecciona un entorno", folders_in_directory)
if selected_folder == "Sin selección":
    st.title("Práctica final No estructurados")
    st.subheader("Javier Álvarez Martínez y David Cocero Quintanilla")
    st.divider()
    st.write("Presentamos aquí nuestra interfaz para permitir a los usuarios explorar nuestro trabajo final de la asignatura que se divide en 2 componentes principales: Texto e Imagen. Para empezar seleccione una de las 2 opciones que aparecen en el desplegable de la izquierda.")
if selected_folder == "Texto":
    st.title("TEXTO")
    st.subheader("Dataset de noticias de la BBC")
    st.write("El conjunto de datos para la generación de resúmenes de textos consta de cuatrocientos diecisiete artículos de 5 temáticas diferentes de la BBC de 2004 a 2005 en la carpeta “News Articles” . Para cada artículo, se proporciona el resumen generado por una persona para cada noticia. Las temáticas que tiene el dataset son: business, entertainment, politics, sport y tech.")
    st.divider()
    st.header("Resumen de las noticias")
    st.write("Hemos implementado diferentes modelos para crear resúmenes de noticias. Para probarlo, empieza elegiendo uno de los tres modelos: con LSTM from Scratch, Modelo de Hugging Face, Few-shot con Hugging Face.")
    st.write("Para medir la eficiencia de los resumenes hemos utilizado BLEU. BLEU es una métrica de evaluación de la calidad de traducción automática que compara un texto generado con uno de referencia, calculando la precisión de las n-gramas coincidentes. Cuanto más alto es el puntaje BLEU, más similar es el texto generado al texto de referencia.")
    modelos = ["LSTM", "Hugging Face","Few-shot"]
    modelo_seleccionado = st.radio("Selecciona un modelo:", modelos)
    st.write("El resumen no puede ser más largo de 200 palabras ")
    txt = st.text_area(
    "Pega aquí la noticia a resumir (tiene que ser en inglés)",height=500
    )
    rsm = st.text_area("Introduce el resumen ")
    st.write(f"La longitud de la noticia es de {len(txt.split())} palabras y la de tu resumen de {len(rsm.split())} " )
    categorias = ["business","entertainment","politics","sport","tech"]
    cat = st.selectbox("Selecciona la categoria de la noticia", categorias)
    if modelo_seleccionado == "Hugging Face":
        btn_gen = st.button("Generar Resumen con modelo de Hugging Face")
        if btn_gen:
            with st.spinner(text="Generando el resumen..."):
                rsm_gen,score = generar_resumen(summarizer,txt,cat,rsm)
            st.subheader("Resultados")
            st.markdown("***Resumen Generado***")
            st.write(rsm_gen)
            st.markdown("***BLEU score obtenido***")
            st.write(f"{score*100:.2f} %")
    if modelo_seleccionado == "Few-shot":
        btn_gen = st.button("Generar Resumen con Modelo de Few-Shot")
        if btn_gen:
            with st.spinner(text="Generando el resumen..."):
                rsm_gen,score = generar_resumen_few_shot(tokenizer,model,txt,rsm)
            st.subheader("Resultados")
            st.markdown("***Resumen Generado***")
            st.write(rsm_gen)
            st.markdown("***BLEU score obtenido***")
            st.write(f"{score*100:.2f} %")
    if modelo_seleccionado == "LSTM":
        btn_gen = st.button("Generar Resumen con Modelo de LSTM")
        if btn_gen:
            with st.spinner(text="Generando el resumen..."):
                rsm_gen,score = generar_resumen_lstm(txt,rsm)
            st.subheader("Resultados")
            st.markdown("***Resumen Generado***")
            st.write(rsm_gen)
            st.markdown("***BLEU score obtenido***")
            st.write(f"{score*100:.2f} %")


    clas = st.button("Prueba nuestro clasificador de noticias")
    if clas:
        modelo, vectorizer= joblib.load('TEXTO/news_class_nb.pkl')
        texto_vectorizado = vectorizer.transform([rsm])
        categoria_predicha = modelo.predict(texto_vectorizado)
        st.write("La categoría predicha es:", categoria_predicha[0])


                
                
               
    
    