#!/usr/bin/env python
# coding: utf-8

import os
import keras.layers
import numpy as np
import streamlit as st
import pandas as pd
import ast
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import keras
from PIL import Image
from keras.models import load_model, Model,Sequential
from transformers import pipeline
from funciones_streamlit import generar_resumen,generar_resumen_few_shot,generar_resumen_lstm,clasificar_imagen_cnn
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,GlobalAveragePooling2D,GlobalMaxPool2D,MaxPooling2D,MaxPool2D,Dropout

def create_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3), filters=32, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=100, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(units=233, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation="softmax"))  # 10 unidades para salida, por las clases

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
@st.cache_resource
def load_resources():
    model_name='google/flan-t5-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    scratch_model = create_model()
    scratch_model.load_weights("IMAGEN\scratch.h5")
   # transfer_model = load_model("IMAGEN\model.transfer_learning_v2.keras")
    return model,tokenizer,summarizer,scratch_model
print(keras.__version__)
model, tokenizer,summarizer,scratch_model = load_resources()
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
    analisis_exploratorio = ["EDA","Modelos"]
    eda = st.radio("Elige si quieres ver un análisis exploratorio del dataset o pasar a probar los modelos. (En cualquier momento puede pulsar el otro botón para ver otra sección)", analisis_exploratorio)
    if eda == "EDA":
        st.header("Análisis exploratorio del dataset (EDA)")
        st.write("El análisis exploratorio de datos es una metodología que permite entender la estructura, patrones y relaciones en un conjunto de datos sin hacer suposiciones previas. Ayuda a identificar tendencias, anomalías y a formular hipótesis para investigaciones más detalladas.")
        st.write("En esta sección, únicamente se hará un recorrido por el análisis exploratorio que hemos realizado, no se incluirá código, úncamente información que consideramos útil de cara a nuestro problema")
        st.subheader("Número de artículos por categoría")
        st.image("imagenesEDA/articulos_por_categoria.png", caption="En la imagen anterior se puede ver el número de artículos dividido por categoría. Este dato nos ha parecido reseñable ya que de cara a nuestro clasificador, tener más cantidad de resúmenes/artículos puede ser importante de cara a una mejor clasificación.", use_column_width=True)
        st.subheader("Número de palabras medias de artículo por categoría")
        st.image("imagenesEDA/palabras_media_noticias.png", caption="Esta imagen es interesante ya que nos dice que noticias esperan un resumen más largo para que de esa forma para los resúmenes que generemos puedan ser de una longitud similar a los existentes", use_column_width=True)
        st.subheader("Número de palabras medias de resúmenes por categoría")
        st.image("imagenesEDA/palabras_media_resumenes.png", caption= "Como hemos comentado nos es muy relevatne poder saber en media cuántas palabras tienen nuestros resúmenes para poder generarlos similar, por ello en la anterior podemos ver de un vistazo el número de palabras medio de los resúmenes por categoría", use_column_width=True)
        st.subheader("Nube de palabras de cada cateogría")
        st.write("Estas nubes de palabras que se ven a continuación, son interesantes para poder saber que tipo de noticias obtendremos buscando por esas palabras. Más adelante, aparece un buscador de noticias usando TF-IDF que se podrá probar y para el que puede ser de ayuda estas nubes de palabras")
        st.image("imagenesEDA/business_wordcloud.png", use_column_width=True)
        st.image("imagenesEDA/entertainment_wordcloud.png", use_column_width=True)
        st.image("imagenesEDA/politics_wordcloud.png", use_column_width=True)
        st.image("imagenesEDA/sport_wordcloud.png", use_column_width=True)
        st.image("imagenesEDA/tech_wordcloud.png", use_column_width=True)
    if eda == "Modelos":
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
                    rsm_gen,score, error = generar_resumen(summarizer,txt,cat,rsm)
                if error:
                    st.warning("¡Cuidado! Introduzca una noticia y un resumen para continuar.")
                else:
                    st.subheader("Resultados")
                    st.markdown("***Resumen Generado***")
                    st.write(rsm_gen)
                    st.markdown("***BLEU score obtenido***")
                    st.write(f"{score*100:.2f} %")
        if modelo_seleccionado == "Few-shot":
            btn_gen = st.button("Generar Resumen con Modelo de Few-Shot")
            if btn_gen:
                with st.spinner(text="Generando el resumen..."):
                    rsm_gen,score, error = generar_resumen_few_shot(tokenizer,model,txt,rsm)
                if error:
                    st.warning("¡Cuidado! Introduzca una noticia y un resumen para continuar.")
                else:
                    st.subheader("Resultados")
                    st.markdown("***Resumen Generado***")
                    st.write(rsm_gen)
                    st.markdown("***BLEU score obtenido***")
                    st.write(f"{score*100:.2f} %")
        if modelo_seleccionado == "LSTM":
            btn_gen = st.button("Generar Resumen con Modelo de LSTM")
            if btn_gen:
                with st.spinner(text="Generando el resumen..."):
                    rsm_gen,score, error = generar_resumen_lstm(txt,rsm)
                if error:
                    st.warning("¡Cuidado! Introduzca una noticia y un resumen para continuar.")
                else:
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

if selected_folder == "Imagen":
    st.title("IMAGEN")
    st.subheader("Dataset de imágenes de 100 deportes")
    st.write("Es una colección de imágenes que cubren 100 deportes diferentes. Las imágenes están en formato jpg con dimensiones de 224x224x3. Los datos están separados en directorios de entrenamiento, prueba y validación. Además, se incluye un archivo CSV para aquellos que deseen usarlo para crear sus propios conjuntos de datos de entrenamiento, prueba y validación.")
    st.divider()
    st.header("Clasificación de imágenes")
    st.write("Presnetamos 2 modelos que buscan realizar una clasificación ")
    modelos = ["CNN from scratch", "Transfer Learning"]
    modelo_seleccionado = st.radio("Selecciona un modelo:", modelos)
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"], help='Arrastra una imagen o haz clic para seleccionarla')  
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen cargada', use_column_width=True)
    opcion_seleccionada = st.selectbox(
        'Seleccione el tipo de imagen que va a introducir:',
        ('Inside city', 'Kitchen', 'Office', 'Store', 'Street', 'Suburb', 'Highway', 'Coast', 'Mountain', 'Open country', 'Industrial', 'Forest', 'Tall building', 'Bedroom', 'Living room')
    )
    if modelo_seleccionado == "CNN from scratch":
        clases = ['hockey', 'tennis', 'baseball', 'swimming', 'polo', 'basketball', 'formula 1 racing', 'boxing', 'football', 'bowling']
        btn_gen = st.button("Clasificar Imagen con el modelo from scratch")
        if btn_gen:
            st.write("Hola")
            with st.spinner(text="Clasificando la imagen"):
                 res = clasificar_imagen_cnn(image, scratch_model)
            st.write(res)
            # if opcion_seleccionada == res:
            #     st.success("¡Correcto!")
            #     st.write(f"Predicción: {res} | imagen introducida: {opcion_seleccionada}", unsafe_allow_html=True)
            # else:
            #     st.error("Incorrecto")
            #     st.write(f"Predicción: {res} | imagen introducida: {opcion_seleccionada}", unsafe_allow_html=True)


                
                
               
    
    