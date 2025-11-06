import cv2, time
import numpy as np
from dataset import Dataset, mean_face, eigenface, classification

start_time = time.time()
yale_classes = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

#vetores de acurácia
ac_yale = []
ac_orl = []

print ("Reading YALE dataset...")
yaleDataset = Dataset("yale")
for test_class in yale_classes:
    yaleDataset.get_images_and_labels(test_class)
    mean_face_yale = mean_face(yaleDataset.images, yaleDataset.max_height, yaleDataset.max_width)
    eigenfaces_yale, mean_images_yale = eigenface(yaleDataset.images, mean_face_yale)
    correct = 0
    accuracy = 0
    for i in range (0, len(yaleDataset.test_images)): #classificar cada imagem do vetor de testes
        image = cv2.resize(yaleDataset.test_images[i], (yaleDataset.max_height, yaleDataset.max_width)).flatten()
        index = classification(image, eigenfaces_yale, mean_images_yale, mean_face_yale)
        if (yaleDataset.labels[index] == yaleDataset.test_labels[i]): #se a imagem pertence a mesma classe da que mais se aproxima
            correct = correct + 1
    #calcular a acurácia
    accuracy = float(correct)/len(yaleDataset.test_images)
    print ("The accuracy for .{} images is {}".format(test_class, accuracy))
    ac_yale.append(accuracy)
sort = np.argsort(ac_yale)
print ("The best image to classify is .{} and the worst is .{}\n".format(yale_classes[sort[len(yale_classes)-1]], yale_classes[sort[0]]))


print ("Reading ORL dataset...")
orlDataset = Dataset("orl")
for test in range(1, 11):
    orlDataset.get_images_and_labels(test)
    mean_face_orl = mean_face(orlDataset.images, orlDataset.max_height, orlDataset.max_width)
    eigenfaces_orl, mean_images_orl = eigenface(orlDataset.images, mean_face_orl)
    correct = 0
    accuracy = 0
    for i in range (0, len(orlDataset.test_images)): #classificar cada imagem do vetor de testes
        image = cv2.resize(orlDataset.test_images[i], (orlDataset.max_height, orlDataset.max_width)).flatten()
        index = classification(image, eigenfaces_orl, mean_images_orl, mean_face_orl)
        if (orlDataset.labels[index] == orlDataset.test_labels[i]): #se a imagem pertence a mesma classe da que mais se aproxima
            correct = correct + 1
    #calcular a acurácia
    accuracy = float(correct)/len(orlDataset.test_images)
    print ("The accuracy for images {} is {}".format((test), accuracy))
    ac_orl.append(accuracy)
print ("Mean accuracy: {}\nStand deviation accuracy: {}\n".format(np.mean(ac_orl), np.std(ac_orl)))
print ("The training and testing took {} seconds".format(time.time() - start_time))
