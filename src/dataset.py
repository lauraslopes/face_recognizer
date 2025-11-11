import cv2, os, time
import numpy as np
from scipy import linalg as LA
from PIL import Image

class Dataset:
    def __init__(self, type):
        self.type = type
        if self.type == 'yale':
            self.path = 'data/yale_faces' # Path to the Yale Dataset
        elif self.type == 'orl':
            self.path = 'data/orl_faces' # Path to the ORL Dataset

        # For face detection we will use the Haar Cascade provided by OpenCV.
        cascadePath = "data/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascadePath)

        self.images = [] # Will contains face images
        self.labels = [] # Will contains the label that is assigned to the image
        self.test_labels = []
        self.test_images = []
        self.max_width = 0
        self.max_height = 0
        self.mean_face = None
        self.mean_images = []
        self.eigenfaces = []

    """
        Append all the absolute image paths in a list image_paths
        We will not read the image with the .sad extension in the training set
        Rather, we will use them to test our accuracy of the training
    """
    def load_images_and_labels(self, test_class):

        image_paths = []
        for root, dirs, files in os.walk(self.path):
            for f in files:
                image_paths.append(os.path.join(root, f))

        for image_path in image_paths:
            # Read the image and convert to grayscale
            image_pil = Image.open(image_path).convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')

            # Get the label of the image
            if self.type == 'yale':
                nbr = int(os.path.split(image_path)[-1].split(".")[0].replace("subject", ""))
                label = os.path.split(image_path)[-1].split(".")[1]
            elif self.type == 'orl':
                nbr = int(os.path.split(image_path)[0].split("/")[-1].replace("s", ""))
                label = int(os.path.split(image_path)[1].split(".")[0])
            
            # Detect the face in the image
            faces = self.faceCascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                self.max_width = max(self.max_width, w)
                self.max_height = max(self.max_height, h)
                # If image class is the same as test class, then append the image to the test images list
                if (label == test_class):
                    self.test_images.append(image[y: y + h, x: x + w])
                    self.test_labels.append(nbr)
                else:
                    self.images.append(image[y: y + h, x: x + w])
                    self.labels.append(nbr)


    def get_mean_face(self):

        # Resize every image
        for i in range(0, len(self.images)):
            self.images[i] = cv2.resize(self.images[i], (self.max_height, self.max_width)).flatten()

            # Calculate mean face
        self.mean_face = np.mean(self.images, axis=0)

    def eigenface(self):
        new_images = [] #Will have the images - mean face
        eigenfaces = []

        #subtrair a face media de todas as imagens no conjunto de imagens
        for i in range (0, len(self.images)):
            new_images.append(self.images[i] - self.mean_face)
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

        self.eigenfaces = eigenfaces
        self.mean_images = new_images

