import numpy as np
from numpy import linalg as LA

def classification(image, yaleDataset):
    distances = []

    #subtrair da imagem teste a face média
    image = image - yaleDataset.mean_face
    projection = np.matmul(yaleDataset.eigenfaces, image) # projecao da imagem que se quer reconhecer
    for i in range (0, len(yaleDataset.mean_images)):
        proj = np.matmul(yaleDataset.eigenfaces, yaleDataset.mean_images[i]) #calcula a projeção de cada imagem do dataset
        distances.append(LA.norm(projection - proj)) #distancia euclidiana das projecoes
    sort = np.argsort(distances) #ordena as distancias

    return sort[0] #retorna o indice da menor distancia