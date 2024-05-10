import nltk
import numpy as np
nltk.download('punkt')
import os
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize
from keras.models import load_model
from tensorflow.keras.layers import Input,Resizing, Rescaling
from tensorflow.keras.models import Model,Sequential
from utility_functions import decode_sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from torchvision import transforms
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.feature_extraction.text import TfidfVectorizer

def search_news(df,request):
    text_content = df['content']
    vector = TfidfVectorizer(max_df=0.3,         # drop words that occur in more than X percent of documents
                                    #min_df=8,      # only use words that appear at least X times
                                    stop_words='english', # remove stop words
                                    lowercase=True, # Convert everything to lower case 
                                    use_idf=True,   # Use idf
                                    norm=u'l2',     # Normalization
                                    smooth_idf=True # Prevents divide-by-zero errors
                                    )
    tfidf = vector.fit_transform(text_content)
    request_transform = vector.transform([request])
    similarity = np.dot(request_transform,np.transpose(tfidf))
    x = np.array(similarity.toarray()[0])
    indices=np.argsort(x)[-3:][::-1]
    return df.loc[indices][["category","content"]]


def clasificar_imagen_cnn(imagen, model):
    img_size = (224, 224)
    
    # Resize and preprocess the image
    img = imagen.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)

    # Map predicted indices to class labels
    class_to_index = {0: 'baseball', 1: 'basketball', 2: 'bowling', 3: 'boxing', 4: 'football', 5: 'formula 1 racing', 6: 'hockey', 7: 'polo', 8: 'swimming', 9: 'tennis'}
    predicted_index = np.argmax(predictions)
    predicted_class = class_to_index[predicted_index]

    return predicted_class

def clasificar_imagen_transfer_learning(imagen, model,carpetas):
    img_size = (299, 299)
    
    # Resize and preprocess the image
    img = imagen.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
   
     # Make predictions
    predictions = model.predict(img_array)
    # Crear el diccionario con los números como clave y los nombres de las carpetas como valor
    class_to_index = {indice: carpeta for indice, carpeta in enumerate(carpetas, start=0)}
    predicted_index = np.argmax(predictions)
    predicted_class = class_to_index[predicted_index]

    return predicted_class

def generar_resumen(summarizer,content,category,summmary_ref):
    score = 0
    error = 0
    try:
        categoria_business = 150
        categoria_tech = 180
        categoria_entertainment = 220
        categoria_politics = 240
        categoria_sport = 180
        if category == 'business':        
            res = summarizer(content, max_length=categoria_business, min_length=100, do_sample=False)
        elif category == 'tech':
            res = summarizer(content, max_length=categoria_tech, min_length=127, do_sample=False)
        elif category == 'entertainment':
            res = summarizer(content, max_length=categoria_entertainment, min_length=150, do_sample=False)
        elif category == 'politics':
            res = summarizer(content, max_length=categoria_politics, min_length=156, do_sample=False)
        elif category == 'sport':
            res = summarizer(content, max_length=categoria_sport, min_length=104, do_sample=False)
        referencias_tokenizadas = [word_tokenize(sent) for sent in summmary_ref.split('. ') if sent]
        resumen_gen = res[0]['summary_text']

        # Tokenizando el resumen generado
        hipotesis_tokenizada = word_tokenize(resumen_gen)

        # Asegurándonos de que las referencias estén en una lista de listas como espera corpus_bleu
        score = corpus_bleu([referencias_tokenizadas], [hipotesis_tokenizada], smoothing_function=SmoothingFunction().method1)
    except:
        error = 1
        
    return resumen_gen,score, error

def generar_resumen_few_shot(tokenizer,model,content,summmary_ref,category):
    error = 0
    score = 0
    with open(f'TEXTO/data/News Articles/{category}/083.txt', 'r') as file:
        News1 = file.read()
    with open(f'TEXTO/data/Summaries/{category}/083.txt', 'r') as file:
        Summary1 = file.read()
    with open(f'TEXTO/data/News Articles/{category}/112.txt', 'r') as file:
        News2 = file.read()
    with open(f'TEXTO/data/Summaries/{category}/112.txt', 'r') as file:
        Summary2 = file.read()
    
    try:
        prompt = f"""You are an expert in news summarization.
    News:
    {News1}
    Summary:
    {Summary1}

    News:
    {News2}
    Summary:
    {Summary2}


    News:
    {content}
    Summary: """
        inputs = tokenizer(prompt, return_tensors='pt')
        output = tokenizer.decode(
            model.generate(
                inputs["input_ids"],
                max_new_tokens=200,
                min_new_tokens=100,
            )[0], 
            skip_special_tokens=True
        )

        referencias_tokenizadas = [word_tokenize(sent) for sent in summmary_ref.split('. ') if sent]
        resumen_gen = output

        # Tokenizando el resumen generado
        hipotesis_tokenizada = word_tokenize(resumen_gen)

        # Asegurándonos de que las referencias estén en una lista de listas como espera corpus_bleu
        score = corpus_bleu([referencias_tokenizadas], [hipotesis_tokenizada], smoothing_function=SmoothingFunction().method1)
    except:
        error = 1
        
        
    return resumen_gen,score, error

def generar_resumen_lstm(content,summmary_ref):
    error = 0
    score = 0
    try:
        encoder_model = load_model("TEXTO/encoder_model.h5")
        decoder_model = load_model("TEXTO/decoder_model.h5")
        resumen_gen = decode_sequence(encoder_model,decoder_model,content)
        referencias_tokenizadas = [word_tokenize(sent) for sent in summmary_ref.split('. ') if sent]
        # Tokenizando el resumen generado
        hipotesis_tokenizada = word_tokenize(resumen_gen)

        # Asegurándonos de que las referencias estén en una lista de listas como espera corpus_bleu
        score = corpus_bleu([referencias_tokenizadas], [hipotesis_tokenizada], smoothing_function=SmoothingFunction().method1)
    except:
        error = 1
        
    return resumen_gen,score, error
    
