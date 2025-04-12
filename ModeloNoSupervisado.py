# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 19:24:46 2025

@author: juanc
K-means es uno de los algoritmos más comunes para clustering. En este caso, intentaremos agrupar las imágenes en dos categorías

"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ruta de las imágenes
ruta_dataset = "frutas"
categorias = ['banana', 'nobanana']

datos = []
nombres_imagenes = []

# Recorrer las imágenes
for categoria in categorias:
    carpeta = os.path.join(ruta_dataset, categoria)
    for archivo in os.listdir(carpeta):
        ruta_img = os.path.join(carpeta, archivo)
        img = cv2.imread(ruta_img)
        if img is None:
            continue
        img = cv2.resize(img, (100, 100))  # Redimensionar para estandarizar
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extraer características: color promedio
        r_mean = np.mean(img_rgb[:, :, 0])
        g_mean = np.mean(img_rgb[:, :, 1])
        b_mean = np.mean(img_rgb[:, :, 2])
        
        datos.append([r_mean, g_mean, b_mean])
        nombres_imagenes.append(ruta_img)

# Convertir a un array de numpy para usar en clustering
X = np.array(datos)

from sklearn.cluster import KMeans

# Número de clústeres (en este caso, 2 para bananas y no bananas)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Ver los clústeres asignados a cada imagen
labels = kmeans.labels_

# Mostrar el resultado de la clasificación
for i in range(len(nombres_imagenes)):
    print(f"Imagen: {nombres_imagenes[i]} - Clúster asignado: {labels[i]}")

# Visualización de los resultados de K-means
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Clustering de frutas")
plt.xlabel("Promedio color R")
plt.ylabel("Promedio color G")
plt.show()

def mostrar_imagenes_por_cluster(labels, nombres_imagenes, X):
    for cluster in set(labels):
        print(f"Imágenes en el clúster {cluster}:")
        #for i in range(len(labels)):
         #   if labels[i] == cluster:
          #      img = cv2.imread(nombres_imagenes[i])
           #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              #  plt.imshow(img_rgb)
              #  plt.axis('off')
              #  plt.show()

# Mostrar imágenes por clúster
mostrar_imagenes_por_cluster(labels, nombres_imagenes, X)
