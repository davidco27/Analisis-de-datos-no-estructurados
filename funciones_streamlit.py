import nltk
import numpy as np
import cv2
nltk.download('punkt')
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

def clasificar_imagen_cnn(imagen, model):
    img_size = (224, 224)
    
    # Resize and preprocess the image
    img = imagen.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Map predicted indices to class labels
    class_to_index = {0: 'baseball', 1: 'basketball', 2: 'bowling', 3: 'boxing', 4: 'football', 5: 'formula 1 racing', 6: 'hockey', 7: 'polo', 8: 'swimming', 9: 'tennis'}
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

def generar_resumen_few_shot(tokenizer,model,content,summmary_ref):
    error = 0
    score = 0

    try:
        prompt = f"""You are an expert in news summarization.
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
    {content}
    Summary: """
        inputs = tokenizer(prompt, return_tensors='pt')
        output = tokenizer.decode(
            model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
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
    
