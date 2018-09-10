#!/usr/bin/python
# -*- coding: latin-1 -*-

# Import the required modules
import cv2, os, time
import numpy as np
from itertools import chain
from scipy import linalg as LA
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

def get_images_and_labels_yale(path, test_class):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    test_labels = []
    test_images = []
    width = []
    height = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        label = os.path.split(image_path)[1].split(".")[1]
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            width.append(w)
            height.append(h)
            # If image class is the same as test class, then append the image to the test images list
            if (label == test_class):
                test_images.append(image[y: y + h, x: x + w])
                test_labels.append(nbr)
            else:
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)

    max_width = max(width)
    max_height = max(height)

    return images, labels, max_width, max_height, test_images, test_labels

def get_images_and_labels_orl(path, test_class):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image 1 in the training set
    # Rather, we will use them to test our accuracy of the training
    # images will contains face images
    images = []
    image_paths = []
    test_labels = []
    test_images = []
    width = []
    height = []
    labels = []
    for d in os.listdir(path):
        directory = os.path.join(path,d)
        image_paths = [os.path.join(directory, f) for f in os.listdir(directory)]

        for image_path in image_paths:
            # Read the image
            image_pil = Image.open(image_path)
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            # Get the label of the image
            nbr = int(os.path.split(image_path)[0].split("/")[1].replace("s", ""))
            class_image = int(os.path.split(image_path)[1].split(".")[0])
            # Detect the face in the image
            faces = faceCascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                width.append(w)
                height.append(h)
                # If image class is the same as test class, then append the image to the test images list
                if (class_image == (test_class+1)): #+1 because images starts on 1.png
                    test_images.append(image[y: y + h, x: x + w])
                    test_labels.append(nbr)
                else:
                    images.append(image[y: y + h, x: x + w])
                    labels.append(nbr)

    max_width = max(width)
    max_height = max(height)

    # return the images list
    return images, labels, max_width, max_height, test_images, test_labels

def mean_face(images, height, width):

	# Resize every image
	for i in range(0, len(images)):
		images[i] = cv2.resize(images[i], (height, width))
		images[i] = images[i].flatten()

    	# Calculate mean face
	mean = np.mean(images, axis=0)

	return mean

def eigenface(images, mean_face):
    new_images = [] #Will have the images - mean face
    eigenfaces = []

    #subtrair a face media de todas as imagens no conjunto de imagens
    for i in range (0, len(images)):
        new_images.append(images[i] - mean_face)
    #calcular matriz de covariancia
    transpose = np.transpose(new_images)
    covariance_matrix = np.matmul(new_images, transpose)
    #calcular autovalores e autovetores
    eigenvals, eigenvecs = LA.eig(covariance_matrix)
    eigenvecs = np.real(eigenvecs)
    index_in_order = np.argsort(eigenvals)
    #pegar os 5 maiores autovalores para calcular as autofaces
    for i in range (1, 6):
        eigenfaces.append(np.matmul(transpose, eigenvecs[index_in_order[len(index_in_order) - i]]))

    #normalizar as autofaces
    for i in range (0, 5):
        eigenfaces[i] = eigenfaces[i] / LA.norm(eigenfaces[i])

    return eigenfaces, new_images

def classification(image, eigenfaces, images, mean_face):
    distances = []

    #subtrair da imagem teste a face média
    image = image - mean_face
    projection = np.matmul(eigenfaces, image) # projecao da imagem que se quer reconhecer
    for i in range (0, len(images)):
        proj = np.matmul(eigenfaces, images[i]) #calcula a projeção de cada imagem do dataset
        distances.append(LA.norm(projection - proj)) #distancia euclidiana das projecoes
    sort = np.argsort(distances) #ordena as distancias

    return sort[0] #retorna o indice da menor distancia

start_time = time.time()
path_yale = 'yale_faces' # Path to the Yale Dataset
path_orl = 'orl_faces' # Path to the ORL Dataset
yale_classes = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
#vetores de acurácia
ac_yale = []
ac_orl = []

# Call the get_images_and_labels functions and get the face images and the
# corresponding labels
print ("Reading YALE dataset...")
for test_class in yale_classes:
    images_yale, labels_yale, width_yale, height_yale, test_images, test_labels = get_images_and_labels_yale(path_yale, test_class)
    mean_face_yale = mean_face(images_yale, height_yale, width_yale)
    eigenfaces_yale, mean_images_yale = eigenface(images_yale, mean_face_yale)
    correct = 0
    accuracy = 0
    for i in range (0, len(test_images)): #classificar cada imagem do vetor de testes
        test_images[i] = cv2.resize(test_images[i], (height_yale, width_yale)).flatten()
        index = classification(test_images[i], eigenfaces_yale, mean_images_yale, mean_face_yale)
        if (labels_yale[index] == test_labels[i]): #se a imagem pertence a mesma classe da que mais se aproxima
            correct = correct + 1
    #calcular a acurácia
    accuracy = float(correct)/len(test_images)
    print ("The accuracy for .{} images is {}".format(test_class, accuracy))
    ac_yale.append(accuracy)
sort = np.argsort(ac_yale)
print ("The best image to classify is .{} and the worst is .{}\n".format(yale_classes[sort[len(yale_classes)-1]], yale_classes[sort[0]]))


print ("Reading ORL dataset...")
for test in range(10):
    images_orl, labels_orl, width_orl, height_orl, images_tests, test_labels = get_images_and_labels_orl(path_orl, test)
    mean_face_orl = mean_face(images_orl, height_orl, width_orl)
    eigenfaces_orl, mean_images_orl = eigenface(images_orl, mean_face_orl)
    correct = 0
    accuracy = 0
    for i in range (0, len(images_tests)): #classificar cada imagem do vetor de testes
        images_tests[i] = cv2.resize(images_tests[i], (height_orl, width_orl)).flatten()
        index = classification(images_tests[i], eigenfaces_orl, mean_images_orl, mean_face_orl)
        if (labels_orl[index] == test_labels[i]): #se a imagem pertence a mesma classe da que mais se aproxima
            correct = correct + 1
    #calcular a acurácia
    accuracy = float(correct)/len(images_tests)
    print ("The accuracy for images {} is {}".format((test+1), accuracy))
    ac_orl.append(accuracy)
print ("Mean accuracy: {}\nStand deviation accuracy: {}\n".format(np.mean(ac_orl), np.std(ac_orl)))
print ("The training and testing took {} seconds".format(time.time() - start_time))
