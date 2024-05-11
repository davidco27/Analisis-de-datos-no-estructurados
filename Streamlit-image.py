#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
import webbrowser
from PIL import Image
from keras.models import Sequential
from funciones_streamlit import clasificar_imagen_cnn,clasificar_imagen_transfer_learning
from keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
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
    scratch_model = create_model()
    scratch_model.load_weights("IMAGEN/scratch.h5")
    transfer_model = create_transfer_model()
    transfer_model.load_weights("IMAGEN/transfer.weights.h5")
    return scratch_model,transfer_model

scratch_model,transfer_model = load_resources()

st.page_link(label="Inicio", page="https://www.marca.com", icon="🌎"),
st.page_link(label="Imagen", page="https://no-estructurados-image.streamlit.app", icon="2️⃣"),
st.page_link(label="Texto", page="https://no-estructurados-texto1.streamlit.app"),

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
        st.image("imagenesIMAGEN/grafiacasScratch.png", use_column_width=True)
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


            
            
            

