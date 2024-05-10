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
from keras.models import Sequential
from transformers import pipeline
from funciones_streamlit import generar_resumen,generar_resumen_few_shot,generar_resumen_lstm,clasificar_imagen_cnn,search_news,clasificar_imagen_transfer_learning
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from keras.layers import MaxPool2D,Dropout

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
def create_transfer_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

    # Congelamos todas las capas menos las 2 ultimas
    for layer in base_model.layers[:-2]:
        layer.trainable = False

    # Clasificador
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.25)(x)
    predictions = Dense(100, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model
@st.cache_resource
def load_resources():
    model_name='google/flan-t5-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    scratch_model = create_model()
    scratch_model.load_weights("IMAGEN/scratch.h5")
    transfer_model = create_transfer_model()
    transfer_model.load_weights("IMAGEN/transfer.weights.h5")
    return model,tokenizer,summarizer,scratch_model,transfer_model

model, tokenizer,summarizer,scratch_model,transfer_model = load_resources()
folders_in_directory = ["Texto","Imagen"]
folders_in_directory.insert(0, "Inicio")
selected_folder = st.sidebar.selectbox("Selecciona un entorno", folders_in_directory)
if selected_folder == "Inicio":
    _,col_img,_,_,_ = st.columns(5)
    with col_img:
        st.image("imagenesDesign\logo-icai.png", width = 300)
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
if selected_folder == "Texto":
    st.title("TEXTO")
    st.subheader("Dataset de noticias de la BBC")
    st.write("El conjunto de datos para la generación de resúmenes de textos consta de 2225 artículos de 5 temáticas diferentes de la BBC de 2004 a 2005 en la carpeta “News Articles” . Para cada artículo, se proporciona el resumen generado por una persona para cada noticia. Las temáticas que tiene el dataset son: business, entertainment, politics, sport y tech.")
    st.write("En esta sección de texto se dispone de un análisis exploratorio del dataset, 3 modelos para la generación de resúmenes y 1 modelo de clasificación de textos")
    analisis_exploratorio = ["EDA","Modelos","Clasificador"]
    eda = st.radio("Elige si quieres ver un análisis exploratorio del dataset, pasar a probar los modelos o probar el clasificador de resúmenes. (En cualquier momento puede pulsar otro botón para ver otra sección)", analisis_exploratorio)
    if eda == "EDA":
        st.divider()
        st.header("Análisis exploratorio del dataset (EDA)")
        st.write("El análisis exploratorio de datos es una metodología que permite entender la estructura, patrones y relaciones en un conjunto de datos sin hacer suposiciones previas. Ayuda a identificar tendencias, anomalías y a formular hipótesis para investigaciones más detalladas.")
        st.write("En esta sección, únicamente se hará un recorrido por el análisis exploratorio que hemos realizado, no se incluirá código, úncamente información que consideramos útil de cara a nuestro problema. Además, al final de la sección se muestra un buscador de noticias por palabras clave basado en TF-IDF")
        st.subheader("Número de artículos por categoría")
        st.image("imagenesEDA/articulos_por_categoria.png", use_column_width=True)
        st.write("En la imagen anterior se puede ver el número de artículos dividido por categoría. Este dato nos ha parecido reseñable ya que de cara a nuestro clasificador, tener más cantidad de resúmenes/artículos puede ser importante de cara a una mejor clasificación.")
        st.subheader("Número de palabras medias de artículo por categoría")
        st.image("imagenesEDA/palabras_media_noticias.png", use_column_width=True)
        st.write("Esta imagen es interesante ya que nos dice que noticias esperan un resumen más largo para que de esa forma para los resúmenes que generemos puedan ser de una longitud similar a los existentes")
        st.subheader("Número de palabras medias de resúmenes por categoría")
        st.image("imagenesEDA/palabras_media_resumenes.png", use_column_width=True)
        st.write("Como hemos comentado nos es muy relevatne poder saber en media cuántas palabras tienen nuestros resúmenes para poder generarlos similar, por ello en la anterior podemos ver de un vistazo el número de palabras medio de los resúmenes por categoría")
        st.subheader("Nube de palabras de cada cateogría")
        st.write("Estas nubes de palabras que se ven a continuación, son interesantes para poder saber que tipo de noticias obtendremos buscando por esas palabras. Más adelante, aparece un buscador de noticias usando TF-IDF que se podrá probar y para el que puede ser de ayuda estas nubes de palabras")
        st.image("imagenesEDA/business_wordcloud.png", use_column_width=True)
        st.image("imagenesEDA/entertainment_wordcloud.png", use_column_width=True)
        st.image("imagenesEDA/politics_wordcloud.png", use_column_width=True)
        st.image("imagenesEDA/sport_wordcloud.png", use_column_width=True)
        st.image("imagenesEDA/tech_wordcloud.png", use_column_width=True)
        st.subheader("Búsqueda de noticias por palabras clave")
        st.write("En este apartado utilizamos TF-IDF para buscar las 5 noticias más relevantes de nuestro dataset basándonos en unas keywords introducidas por el usuario.")
        st.write("""TF-IDF (Term Frequency-Inverse Document Frequency) es una técnica utilizada en procesamiento de lenguaje natural que asigna un peso a cada término en un documento basado en dos factores:
La frecuencia del término en el documento (TF), lo que indica su importancia relativa en el documento.
La inversa de la frecuencia del término en el conjunto de documentos (IDF), lo que reduce el peso de los términos comunes en todo el conjunto de documentos.
Esto permite representar documentos como vectores numéricos, donde los términos más relevantes tienen un peso más alto, lo que es útil para tareas como la clasificación de textos y la recuperación de información.""")
        df = pd.read_csv('TEXTO/news_df.csv')
        request = st.text_input("Introducir de 1 a 3 palabras clave separadas por un espacio entre cada una")
        btn_search = st.button("Buscar noticias")
        if btn_search:
              with st.spinner(text="Seleccionando las noticias..."):
                 st.table(search_news(df,request))
    if eda == 'Clasificador':
        st.divider()
        st.header("Clasificador de resúmenes")
        st.write("En esta sección hemos creado un clasificador de noticias basado en Naive Bayes. La clasificación se realiza en base a las clases que teníamos definidas y hemos explicado ya.")
        st.write("En primer lugar, aprovechamos las funcionalidades de TF-IDF para transformar los textos en vectores y de esa forma poder entrenar al clasificador.")
        st.write(""" Un clasificador de textos basado en Naive Bayes, como MultinomialNB(), que es el que se ha usado en este caso, es un algoritmo de aprendizaje supervisado utilizado para clasificar textos en categorías o etiquetas específicas. Este clasificador asume que las características (en este caso, las palabras) son independientes entre sí, lo que simplifica el cálculo de la probabilidad condicional. MultinomialNB() específicamente se utiliza cuando las características son frecuencias de ocurrencia de palabras y se asume una distribución multinomial de estas características. Durante el entrenamiento, el algoritmo calcula la probabilidad de que una palabra pertenezca a una clase dada utilizando el teorema de Bayes. Luego, durante la predicción, el clasificador utiliza estas probabilidades para determinar la clase más probable para un nuevo documento o texto de entrada. Este enfoque es eficiente y efectivo para clasificar textos en categorías predefinidas, como análisis de sentimientos, detección de spam o categorización de noticias.""")
        st.write("Los resultados para las diferentes clases obtenidas con el clasificador son:")
        st.image("imagenesEDA/TestClasificacion.png", use_column_width=True)
        st.write("La matriz de clasificación muestra un rendimiento sólido con una precisión, recall y f1-score promedio de 0.93. Las categorías business y tech destacan por su precisión y recall altos (alrededor de 0.95), mientras que entertainment presenta un recall más bajo (0.85). Politics tiene una precisión relativamente menor (0.84), mientras que sport destaca con un recall alto (0.99). En general, el modelo mantiene una precisión constante en todas las categorías con un f1-score promedio de 0.93")
        st.write("Además, hemos querido hacer un test con resúmenes generados usando HuggingFace, uno de los modelos disponibles para probar. Hemos generado 15 resúmnes por categoría y los resultados obtenidos son los siguientes:")
        st.image("imagenesEDA/TestClasificacion_HF.png", use_column_width=True)
        st.write("Las categorías business, entertainment y sport alcanzan una precisión y f1-score perfectos o casi perfectos (0.97-1.00). Politics tiene una precisión ligeramente más baja (0.83) pero compensa con un recall perfecto (1.00). Tech presenta una ligera caída en recall (0.88). En general, el modelo demuestra un alto nivel de consistencia con un f1-score promedio de 0.96.")
        rsm = st.text_area("Introduce el resumen para clasificar")
        categorias = ["business","entertainment","politics","sport","tech"]
        cat = st.selectbox("Selecciona la categoria de la noticia", categorias)
        st.write("Para clasificar el resumen previamente introducido pulse el siguiente botón:")
        bt = st.button("CLASIFICAR")
        if bt:
            modelo, vectorizer= joblib.load('TEXTO/news_class_nb.pkl')
            texto_vectorizado = vectorizer.transform([rsm])
            categoria_predicha = modelo.predict(texto_vectorizado)
            if cat == categoria_predicha[0]:
                st.success("¡Correcto!")
                st.write(f"Predicción: {categoria_predicha[0]} | categoría introducida: {cat}", unsafe_allow_html=True)
            else:
                st.error("Incorrecto")
                st.write(f"Predicción: {categoria_predicha[0]} | categoría introducida: {cat}", unsafe_allow_html=True)

    if eda == "Modelos":
        st.divider()
        st.header("Resumen de las noticias")
        st.write("Hemos implementado diferentes modelos para crear resúmenes de noticias. Para probarlo, empieza elegiendo uno de los tres modelos: con arquitectura Encoder-Decoder from Scratch, Modelo de Hugging Face, Few-shot con Hugging Face.")
        st.write("Para medir la eficiencia de los resumenes hemos utilizado BLEU. BLEU es una métrica de evaluación de la calidad de traducción automática que compara un texto generado con uno de referencia, calculando la precisión de las n-gramas coincidentes. Cuanto más alto es el puntaje BLEU, más similar es el texto generado al texto de referencia.")
        modelos = ["Encoder-Decoder", "Hugging Face","Few-shot"]
        modelo_seleccionado = st.radio("Selecciona un modelo:", modelos)
        txt = st.text_area(
        "Pega aquí la noticia a resumir (tiene que ser en inglés)",height=500
        )
        st.write("El resumen no puede ser más largo de 200 palabras ")
        rsm = st.text_area("Introduce el resumen ")
        st.write(f"La longitud de la noticia es de {len(txt.split())} palabras y la de tu resumen de {len(rsm.split())} " )
        categorias = ["business","entertainment","politics","sport","tech"]
        cat = st.selectbox("Selecciona la categoria de la noticia", categorias)
        if modelo_seleccionado == "Hugging Face":
            st.write("En este modelo de texto, se ha utilizado un modelo preentrenado de HuggingFace preparado para resumir. Para seleccionar la longitud de los resúmenes lo que hemos hecho es utilizar como máximo las longitudes medias de resúmenes que teníamos en los datos de partida y como valor mínimo de longitud hemos usado el máximo menos 30 aproximadamente.")
            st.write("""El modelo "facebook/bart-large-cnn" es parte de la familia BART (BART: Bidirectional and Auto-Regressive Transformers), desarrollado por Facebook AI. BART es un modelo basado en la arquitectura Transformer que ha demostrado ser efectivo en tareas de generación de lenguaje, traducción, resumen de texto...""")
            st.write("""BART es un modelo transformer encoder-encoder (seq2seq) con un encoder bidireccional (similar a BERT) y un decoder autoregresivo (similar a GPT). BART se pre-entrena mediante la corrupción de texto con una función de ruido arbitraria y el aprendizaje de un modelo para reconstruir el texto original. """)
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
            st.write("En este modelo de generación de resúmenes se utilizado un modelo de ámbito general al que se prepara para generar resúmenes.")
            st.write(""" El modelo elegido es: "google/flan-t5-base", un modelo de lenguaje basado en la arquitectura T5, desarrollado por Google, con un tamaño de base. Este modelo se puede utilizar para una amplia gama de tareas de procesamiento de lenguaje natural, como traducción, generación de texto, respuesta a preguntas, resumen de texto, entre otras. En este caso lo que se utiliza es el modelo de base y mediante prompt se le hace un few shot.""")
            st.write("Few-shot learning es un enfoque de aprendizaje automático donde un modelo se entrena con solo un pequeño número de ejemplos de entrenamiento por clase o tarea, lo que le permite generalizar a nuevas clases o tareas con un número limitado de ejemplos de prueba.")
            st.write("El ejemplo de prompt sería como el siguiente (En las pruebas hemos usado una categoría definida, para la app se ha generalizado y el prompt depende de la categoría de artículo):")
            st.code(""" You are an expert in news summarization.
News:

BMW to recall faulty diesel cars

BMW is to recall all cars equipped with a faulty diesel fuel-injection pump supplied by parts maker Robert Bosch.

The faulty part does not represent a safety risk and the recall only affects pumps made in December and January. BMW said that it was too early to say how many cars were affected or how much the recall would cost. The German company is to extend a planned production break at one of its plants due to the faulty Bosch part. The Dingolfing site will now be closed all next week instead of for just two days. The additional three-day stoppage will mean a production loss of up to 3,600 vehicles, BMW said, adding that it was confident it could make up the numbers later.

Bosch has stopped production of the part but expects to restart by 2 February. The faulty component does not represent a safety risk but causes the motor to stall after a significant amount of mileage. When asked if BMW would be seeking compensation from Bosch, the carmaker's chief executive Helmut Panke said: "we will first solve the problem before talking about who will pay". Audi and Mercedes Benz were also supplied with the defective diesel fuel-injection pumps but neither of them have to recall any vehicles. A spokesman for DaimlerChrysler, parent company of Mercedes Benz, said it will however have to halt some production. It is to close the Mercedes factory in Sindelfingen on Monday and Tuesday. Audi said it had been hit by production bottlenecks, due to a shortage of unaffected Bosch parts.


Summary:
BMW is to recall all cars equipped with a faulty diesel fuel-injection pump supplied by parts maker Robert Bosch.The German company is to extend a planned production break at one of its plants due to the faulty Bosch part.The faulty part does not represent a safety risk and the recall only affects pumps made in December and January.Audi said it had been hit by production bottlenecks, due to a shortage of unaffected Bosch parts.A spokesman for DaimlerChrysler, parent company of Mercedes Benz, said it will however have to halt some production.Audi and Mercedes Benz were also supplied with the defective diesel fuel-injection pumps but neither of them have to recall any vehicles.


You are an expert in news summarization.
News:

US gives foreign firms extra time

Foreign firms have been given an extra year to meet tough new corporate governance regulations imposed by the US stock market watchdog.

The Securities and Exchange Commission has extended the deadline to get in line with the rules until 15 July 2006. Many foreign firms had protested that the SEC was imposing an unfair burden. The new rules are the result of the Sarbanes-Oxley Act, part of the US clean-up after corporate scandals such as Enron and Worldcom. Section 404 of the Sox Act, as the legislation is nicknamed, calls for all firms to certify that their financial reporting is in line with US rules. Big US firms already have to meet the requirements, but smaller ones and foreign-based firms which list their shares on US stock markets originally had until the middle of this year.

Over the past few months, delegations of European and other business leaders have been heading to the SEC's Washington DC headquarters to protest. They say the burden is too expensive and the timescale too short and some, particularly the UK's CBI, warned that companies would choose to let their US listings drop rather than get in line with section 404. The latest delegation from the CBI met SEC officials on Wednesday, just before the decision to relax the deadline was announced. "I think this signifies a change of heart at the SEC," CBI director-general Sir Digby Jones told the BBC's Today programme. "They have been listening to us and to many overseas companies, who have reminded America what globalisation really means: that they can't make these rules in isolation." The SEC said it had taken into consideration the fact that foreign companies were already working to meet more onerous financial reporting rules in their home countries. The European Union, in particular, was imposing new international financial reporting standards in 2005, it noted. "I don't underestimate the effort (compliance) will require... but this extension will provide additional time for those issuers to take a good hard look at their internal controls," said Donald Nicolaisen, the SEC's chief accountant.


Summary:
Many foreign firms had protested that the SEC was imposing an unfair burden.The SEC said it had taken into consideration the fact that foreign companies were already working to meet more onerous financial reporting rules in their home countries.Section 404 of the Sox Act, as the legislation is nicknamed, calls for all firms to certify that their financial reporting is in line with US rules.Foreign firms have been given an extra year to meet tough new corporate governance regulations imposed by the US stock market watchdog.Big US firms already have to meet the requirements, but smaller ones and foreign-based firms which list their shares on US stock markets originally had until the middle of this year.The European Union, in particular, was imposing new international financial reporting standards in 2005, it noted.



News:

Australia rates at four year high

Australia is raising its benchmark interest rate to its highest level in four years despite signs of a slowdown in the country's economy.

The Reserve Bank of Australia lifted interest rates 0.25% to 5.5%, their first upwards move in more than a year. However, shortly after the Bank made its decision, new figures showed a fall in economic growth in the last quarter. The Bank said it had acted to curb inflation but the move was criticised by some analysts.

The rate hike was the first since December 2003 and had been well-flagged in advance. However, opposition parties and some analysts said the move was ill-timed given data showing the Australian economy grew just 0.1% between October and December and 1.5% on an annual basis.

The figures, representing a decline from the 0.2% growth in GDP seen between July and September, were below market expectations. Consumer spending remains strong, however, and the Bank is concerned about growing inflationary pressures. "Over recent months it has become increasingly clear that remaining spare capacity in the labour and goods markets is becoming rather limited," said Ian Macfarlane, Governor of the Reserve Bank.

At 2.6%, inflation remains within the Bank's 2-3% target range. However, exports declined in the second half of 2004, fuelling a rise in the country's current account deficit - the difference in the value of imports compared to exports - to a record Australian dollar 29.4bn. The Australian government said the economy remained strong with unemployment at a near 30 year low. "The economy has been strong and it is properly moderating but it doesn't look to me like it's slowing in any unreasonable way," said Treasurer Peter Costello. Stock markets had factored in the likelihood of a rate rise but analysts still expressed concern about the strength of the economy. "That 1.5% annual growth rate is the lowest we have seen since the post-election slump we saw back in 2000-1," said Michael Blythe, chief economist at the Commonwealth Bank of Australia. "This suggests the economy really did slow very sharply in the second half of 2004."

Summary: """)
            st.write("Como se puede ver, primero se le pone en la situación de que es un expero en hacer resúmenes, esto es un best practice de prompt engineering (Prompt engineering implica la creación de preguntas o ejemplos específicos para guiar a un modelo de lenguaje a producir respuestas deseadas o realizar tareas específicas, lo que puede mejorar su rendimiento y adaptabilidad en diversas aplicaciones de procesamiento del lenguaje natural). Posteriormente se pasan 3 ejemplos de resúmenes para que tenga un baseline de como comportarse. Finalmente se pide que genere el resumen. En este caso la generación de resumen se había limitado a 50 tokens para las pruebas ya que los recursos eran limitados, en la aplicación se ha puesto a 200 nuevos tokens")
            st.write("A continuación se muestra una imagen con el resumen que venia en los datos y el resumen generado:")
            st.image("imagenesEDA/EjemploFewShot.png", use_column_width=True)
            st.write("Como se puede ver el resúmen generado es razonable teniendo en cuenta la limitación de caracteres.")
            btn_gen = st.button("Generar Resumen con Modelo de Few-Shot")
            if btn_gen:
                with st.spinner(text="Generando el resumen..."):
                    rsm_gen,score, error = generar_resumen_few_shot(tokenizer,model,txt,rsm,cat)
                if error:
                    st.warning("¡Cuidado! Introduzca una noticia y un resumen para continuar.")
                else:
                    st.subheader("Resultados")
                    st.markdown("***Resumen Generado***")
                    st.write(rsm_gen)
                    st.markdown("***BLEU score obtenido***")
                    st.write(f"{score*100:.2f} %")
        if modelo_seleccionado == "Encoder-Decoder":
            st.write("Este modelo tiene como objetivo el resumen de noticias, pero para su consecución utiliza una estrategia completamente distinta a los otros dos. En vez de emplear un modelo de LLM pre-entrenado, usa una arquitectura de encoder-decoder from scratch para tratar problemas de tipo Seq2seq.")
            st.write("""La primera parte de la red es el encoder. En esta sección se recibe la noticia tokenizada y se crean los Embeddings. Estos embeddings se pasan a continuación por 3 redes LSTM para obtener un vector de tamaño fijo con los encodings.""")
            st.code("""encoder_inputs = Input(shape=(max_text_len,))

# Capa de embeddings
enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

#Lstm 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#Lstm 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#Lstm 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)""")
            st.write("El siguiente componente es el decoder que recibe como entrada la salida del encoder. Crea los embedings de los resumenes y los introduce en una LSTM. Un detalle clave del encoder es la Attention Layer que busca recoger conexiones entre distintos tokens para mejorar el rendimiento. Se concatena la salida de la LSTM y de la Attention Layer y se pasan a una red Dense que genera los tokens de salida junto con sus probabilidades.")
            st.code("""
decoder_inputs = Input(shape=(None,))


dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# Attention layer
attn_layer = AttentionLayer(name='attention_layer') 
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

# Dense Layer
decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)
""")
            st.write("Se fija el tamaño máximo de las noticias de entrada a 350. Como todas las entradas al modelo tienen que tener el mismo tamaño, se le añade padding al final de las noticias para llegar al máximo.")
            st.write("Como se puede comprobar, la salida del modelo es una secuencia en la que se repite el mismo token una y otra vez. El motivo es que las secuencias de entrada son demasiado largas y complejas, además que para un entrenamiento desde cero se necesitan más de las aproximadamente 2000 muestras que tenemos. Esta arquitectura es bastante limitada en un problema así y no puede competir con los otros dos modelos de Hugging Face que presentamos.")
            btn_gen = st.button("Generar Resumen con Modelo from Scratch")
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

if selected_folder == "Imagen":
    st.title("IMAGEN")
    st.subheader("Dataset de imágenes de 100 deportes")
    st.write("Es una colección de imágenes que cubren 100 deportes diferentes. Las imágenes están en formato jpg con dimensiones de 224x224x3. Los datos están separados en directorios de entrenamiento, prueba y validación.")
    st.divider()
    img_pos = ["Generador","Clasificador"]
    img = st.radio("Elige si quieres probar el generador de imágenes o el clasificador de imágenes por deporte. (En cualquier momento puede pulsar otro botón para ver la otra sección)", img_pos)
    if img == "Generador":
        st.header("Generador de imágenes con GAN")
        st.write("""Una de las features que hemos implementado con este dataset es la generación de imágenes artificiales utilizando una red Generativa Adversaria (GAN).
                La GAN consiste en dos redes neuronales compitiendo entre sí. 
                La primera red, llamada generador, crea imágenes a partir de ruido aleatorio que intentan pasar como reales; mientras que la segunda red, el discriminador, 
                evalúa estas imágenes y decide si son reales o falses. A medida que pasa el tiempo el generador mejora su capacidad para engañar al discriminador,
                y el discriminador mejora su habilidad para distinguir lo real de lo generado. De esta forma el sistema como un todo aprende a generar datos cada vez más realistas.
                """)
        st.write("Ponemos a entrenar la GAN durante 50 épocas con imágenes de los 100 deportes, mostrando en cada época 4 imágenes de la GAN para ver cómo evoluciona. Al principio las imágenes de la GAN no es más que el ruido que le metemos:")
        st.image("imagenesIMAGEN/ganStart.png", use_column_width=True)
        st.write("A medida que se van ajustando los pesos de la red con el paso de las épocas, se empiezan a perfilar formas borrosas. Ya no es ruido aleatorio, sino que se parece más a una posible imagen, como las que vemos aquí correspondientes a las épocas 12 y 13")
        st.image("imagenesIMAGEN/ganMedium.png", use_column_width=True)
        st.write("Según avanza el entrenamiento esperamos que la calidad mejore bastante, acercandonos a imágenes que parezcan relativamente reales. Sin embargo, parece que llega a un punto donde el generador se estanca y no puede mejorar casi. Llegamos al final del entrenamiento con unas imágenes similares a las que nos encontramos en las épocas 12 y 13:")
        st.image("imagenesIMAGEN/ganEnd.png", use_column_width=True)
        st.write("El decepcionante resultado obtenido por la GAN se debe a la gran diversidad de las imágenes usadas para el entrenamiento, que hace que el generador se confunda constantemente y no pueda aprender a generar imágenes reales. Una posible solución sería encontrar un dataset grande de imágenes deportivas más parecidas entre sí.")
    if img == "Clasificador":    
        st.header("Clasificación de imágenes")
        st.write("Hemos creado una funcionalidad que permite al usuario subir una imagen y te devuelve el deporte al que se corresponde. Para ello presentamos 2 modelos distintos: el primero es un modelo from scratch que implementa una CNN mientras que el segundo utiliza Transfer Learning.")
        st.write("Para cada modelo primero hay una parte para subir una imagen y la categoría de deporte a la que pertence, a continuación, una pequeña explicación y al final del todo existe la posibilidad de hacer una prueba.")
        st.write("Para realizar una prueba, elige una de las 2 opciones y sube una imagen en formato JPEG")
        modelos = ["CNN from scratch", "Transfer Learning"]
        modelo_seleccionado = st.radio("Selecciona un modelo:", modelos)
        if modelo_seleccionado == "CNN from scratch":
            st.write("Para el modelo from scratch, debido a su complejidad y a la capacidad de cómputo requerida hemos decidido limitarlo a 10 deportes que son:")
            st.markdown("""
            * hockey
            * tennis
            * baseball
            * swimming
            * polo
            * basketball
            * formula 1 racing
            * boxing
            * football
            * bowling
            """)
            uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"], help='Arrastra una imagen o haz clic para seleccionarla')  
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Imagen cargada', use_column_width=True)
            clases = ['hockey', 'tennis', 'baseball', 'swimming', 'polo', 'basketball', 'formula 1 racing', 'boxing', 'football', 'bowling']
            opcion_seleccionada = st.selectbox(
            'Seleccione el tipo de imagen que va a introducir:',
            clases
        )
            st.write("La CNN toma como entrada imágenes de tamaño 224x224x3 y utiliza 4 capas para extraer features,todas con filtros de 3x3. El número de filtros utilizados en cada capa es respectivamente: 32,100,64 y 128. Después de cada capa de convolución con activación RELU, se utiliza una capa de MaxPooling para reducir la dimensionalidad espacial de la salida. Además se incorporan capas de Dropout para evitar el sobreajuste del modelo. Finalmente, se aplana la salida y se conecta a dos fully connected layers,con una capa de salida con activación softmax para generar las probabilidades de pertenencia a cada una de las 10 clases.")
            st.write("Para el caso de la CNN, como se ha mencionado previamente únicamente se han seleccionado 10 deportes, por ello únicamente se puede elegir entre esos deportes. Esto se ha hecho así para que sea posible el entrenamiento y un acierto razonable.")
            st.write("Para entrenar la CNN se ha utilizado una herramienta conocida como optuna. Como los recursos de los que se dispone para entrenar y optimizar son limitados. En la etapa de optimización se ha buscado únicamente los parámetros del número de filtros de la 2 capa convolucional y el número de neuronas de la primera capa densa. Los valores obtenidos son: Número de filtros: 100, Número de neuronas: 233")
            st.write("El código de la red finalmente sería el siguiente:")
            st.code("""
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
                    """)
            st.write("El summary del modelo para ver el número de parámetros y de un vistazo las capas es:")
            st.image("imagenesIMAGEN/model_summary.png", use_column_width=True)
            st.write("Para terminar con la explicación, los resultados obtenidos con el modelo en la fase de entrenamiento son los siguientes:")
            st.image("imagenesIMAGEN\grafiacasScratch.png", use_column_width=True)
            st.write("En las gráficas se puede apreciar el entrenamiento del modelo a lo largo de las épocas. Cabe destacar una divergencia significativa entre la precisión y la loss de entrenamiento (líneas azules) frente a la precisión y la loss de validación (líneas rojas). El modelo alcanza una alta precisión en los datos de entrenamiento, pero la fluctuante y decreciente precisión en los datos de validación, junto con la divergencia en las curvas de pérdida, sugieren un claro caso de sobreajuste.")
            st.write("Vamos a analizar la precisión para las diferentes clases en test")
            st.image("imagenesIMAGEN/testScratch.png", use_column_width=True)
            st.write("El informe de clasificación y las gráficas de precisión y loss indican que el modelo presenta un caso claro de sobreajuste, donde la precisión en los datos de entrenamiento es alta, pero la precisión en validación fluctúa y se mantiene baja, especialmente para clases como bowling y tennis. La discrepancia entre precisión y recall para algunas categorías refleja problemas de desequilibrio y sesgo en los datos. Este clasificador tiene un desbalanceo en la precisión entre clases. Entendemos que es así porque es un modelo sencillo, únicamente para probar a realizar el modelo from scratch y pegarnos con ello.")
            st.write("PULSAR EL BOTÓN DE DEBAJO PARA PROBAR EL CLASIFICADOR FROM SCRATCH")
            btn_gen = st.button("Clasificar Imagen con el modelo from scratch")
            if btn_gen:
                if 'image' not in locals():
                    st.warning("¡Cuidado! Tiene que subir primero la imagen a clasificar.")
                else:
                    with st.spinner(text="Clasificando la imagen"):
                        res = clasificar_imagen_cnn(image, scratch_model)
                    if opcion_seleccionada == res:
                        st.success("¡Correcto!")
                        st.write(f"Predicción: {res} | imagen introducida: {opcion_seleccionada}", unsafe_allow_html=True)
                    else:
                        st.error("Incorrecto")
                        st.write(f"Predicción: {res} | imagen introducida: {opcion_seleccionada}", unsafe_allow_html=True)
        if modelo_seleccionado == "Transfer Learning":
            uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"], help='Arrastra una imagen o haz clic para seleccionarla')  
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Imagen cargada', use_column_width=True)
            directorio = 'IMAGEN/data/train'
            carpetas = [nombre for nombre in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, nombre))]
            carpetas.sort()
            opcion_seleccionada = st.selectbox(
            'Seleccione el tipo de imagen que va a introducir:',
            carpetas
        )           
            st.write("En este caso hemos utilizado como modelo base la red neuronal EfficientNetB0, con los pesos de Imagenet. Descongelamos las últimas dos capas del modelo base y añadimos el clasificador al final. A continuación reentrenamos el modelo con nuestro dataset de deportes buscando mejorar la accuracy.")
            st.write("Vamos a comentar los resultados obtenidos durante el entrenamiento con el modelo. En este caso sí hemos podido usar las 100 clases porque partíamos de un modelo base muy complejo que hemos adaptado a nuestras necesidades")
            st.write("los resultados obtenidos con el modelo en la fase de entrenamiento son los siguientes:")
            st.image("imagenesIMAGEN/grafiacasTransfer.png", use_column_width=True)
            st.write("La rápida disminución de la pérdida de entrenamiento indica que el modelo aprende bien. La pérdida de validación se mantiene estable, lo que sugiere que el modelo generaliza bien, aunque el ligero aumento al final podría ser un signo de sobreajuste. El modelo muestra un rendimiento consistente con una alta precisión tanto en el conjunto de entrenamiento como en el de validación.")
            st.write("Vamos a analizar la precisión para las diferentes clases en test. Como son 100 deportes el resumen es largo, por ello se ha dividido en 3 imágenes. Si la visualiación es complicada, se puede ampliar la imagen en la esquina superior derecha de cada una.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image("imagenesIMAGEN/testLearning1.png", use_column_width=True)
            with col2:
                st.image("imagenesIMAGEN/testLearning2.png", use_column_width=True)
            with col3:
                st.image("imagenesIMAGEN/testLearning3.png", use_column_width=True)
            st.write("El modelo logra un rendimiento global sobresaliente, con una precisión general del 97% y un promedio ponderado de F1-score del 96%. Algunas clases como basketball y trapeze presentan un menor rendimiento, posiblemente debido a la limitada cantidad de datos de entrenamiento. La mayoría de las clases tienen un rendimiento perfecto, lo que sugiere que el modelo puede diferenciar con precisión entre la mayoría de las categorías. El modelo logra una alta precisión y recall en la clasificación de la mayoría de las clases, con un promedio ponderado de F1-score del 96%, indicando un excelente rendimiento.")
            st.write("PULSAR EL BOTÓN DE DEBAJO PARA PROBAR EL CLASIFICADOR CON TRANSFER LEARNING")
            btn_gen = st.button("Clasificar Imagen con el modelo de Transfer Learning")
            if btn_gen:
                if 'image' not in locals():
                    st.warning("¡Cuidado! Tiene que subir primero la imagen a clasificar.")
                else:
                    with st.spinner(text="Clasificando la imagen"):
                        res = clasificar_imagen_transfer_learning(image, transfer_model,carpetas)
                    if opcion_seleccionada == res:
                        st.success("¡Correcto!")
                        st.write(f"Predicción: {res} | imagen introducida: {opcion_seleccionada}", unsafe_allow_html=True)
                    else:
                        st.error("Incorrecto")
                        st.write(f"Predicción: {res} | imagen introducida: {opcion_seleccionada}", unsafe_allow_html=True)


                
                
               
    
    