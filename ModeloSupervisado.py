# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 18:59:25 2025

@author: juanc
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

# Ruta base
ruta_dataset = "frutas"
categorias = ['banana', 'nobanana']

datos = []

for categoria in categorias:
    carpeta = os.path.join(ruta_dataset, categoria)
    for archivo in os.listdir(carpeta):
        ruta_img = os.path.join(carpeta, archivo)
        img = cv2.imread(ruta_img)
        if img is None:
            continue
        img = cv2.resize(img, (100, 100))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extraer características: color promedio RGB
        r = np.mean(img_rgb[:, :, 0])
        g = np.mean(img_rgb[:, :, 1])
        b = np.mean(img_rgb[:, :, 2])

        datos.append([r, g, b, categoria])

# Crear DataFrame
df = pd.DataFrame(datos, columns=['R', 'G', 'B', 'etiqueta'])

# Codificar etiquetas
df['etiqueta_binaria'] = df['etiqueta'].apply(lambda x: 1 if x == 'banana' else 0)

X = df[['R', 'G', 'B']]
y = df['etiqueta_binaria']

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo árbol de decisión
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(export_text(clf, feature_names=['R', 'G', 'B']))

def clasificar_imagen(ruta_img):
    img = cv2.imread(ruta_img)
    img = cv2.resize(img, (100, 100))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    r = np.mean(img_rgb[:, :, 0])
    g = np.mean(img_rgb[:, :, 1])
    b = np.mean(img_rgb[:, :, 2])
    
   
    entrada = pd.DataFrame([[r, g, b]], columns=['R', 'G', 'B'])
    pred = clf.predict(entrada)

    return "banana" if pred[0] == 1 else "no banana"

# Ejemplo de uso
print(clasificar_imagen("C:/Users/juanc/anaconda3/Lib/site-packages/spyder_kernels/EjemploApredizajeSpuervisado/bananasnousadasporelsistema/download.jpg"))

