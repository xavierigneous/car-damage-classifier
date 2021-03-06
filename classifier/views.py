from urllib import response
from django.shortcuts import render, HttpResponse
from django.http import HttpResponse, FileResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from PIL import Image
#import time, sqlalchemy, os
#from sqlalchemy.exc import SQLAlchemyError
import json, base64, os, time
import numpy as np
import pandas as pd
from io import BytesIO
#import psycopg2 as pg
import keras
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "damage_classify"))

damage=['Damage', 'No Damage']
location=['Front','Rear','Side']
img_save_path = r'\temp_data'
def home(request):
    if 'image_upload' in request.POST and request.method == "POST":
        uploaded_image=request.FILES['file']
        print(uploaded_image.name)
        #image=uploaded_image
        display_image = base64.b64encode(uploaded_image.read()).decode('utf-8')
        
        with open(img_save_path+uploaded_image.name, 'wb+') as f:
            f.write(base64.b64decode(display_image))
        
        img1=keras.preprocessing.image.load_img(img_save_path+uploaded_image.name, target_size=(300, 300))
        x = keras.preprocessing.image.img_to_array(img1) / 255
        x = np.expand_dims(x, axis=0)
        damage_model = keras.models.load_model(os.path.join(os.getcwd(),r'classifier\models\MobileNet_Car_Classifier.h5'))
        location_pred_model = keras.models.load_model(os.path.join(os.getcwd(),r'classifier\models\MobileNet_Car_Damage_Location.h5'))
        damage_incured = damage[int(damage_model.predict(x).argmax(axis=-1))]
        print(damage_incured)   
        damage_report='{} Incurred'.format(damage_incured)
        if damage_incured=='Damage':
            img2=keras.preprocessing.image.load_img(img_save_path+uploaded_image.name, target_size=(224, 224))
            x = keras.preprocessing.image.img_to_array(img2) / 255
            x = np.expand_dims(x, axis=0)
            location_damage = location[int(location_pred_model.predict(x).argmax(axis=-1))]
            damage_report='{} Incurred in the {}'.format(damage_incured, location_damage)
            messages.error(request,damage_report,extra_tags='damaged')
        else:
            messages.success(request,damage_incured,extra_tags='not_damaged')    
        
        response = {'uploaded_image':display_image,'damage_report':damage_report}
        return render(request, 'homepage.html', response)
    else:
        return render(request, 'homepage.html')