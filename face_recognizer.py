import cv2, time
import numpy as np
from src.dataset import Dataset
from src.classify import classification

def execute(type, classes): 
    print ("Reading {} dataset...".format(type.upper()))
    accuracy = []
    dataset = Dataset(type)
    for test_class in classes:
        dataset.load_images_and_labels(test_class)
        dataset.get_mean_face()
        dataset.eigenface()
        correct = 0
        acc = 0
        for i in range (0, len(dataset.test_images)): #classificar cada imagem do vetor de testes
            image = cv2.resize(dataset.test_images[i], (dataset.max_height, dataset.max_width)).flatten()
            index = classification(image, dataset)
            if (dataset.labels[index] == dataset.test_labels[i]): #se a imagem pertence a mesma classe da que mais se aproxima
                correct = correct + 1
        #calcular a acurácia
        acc = float(correct)/len(dataset.test_images)
        print ("The accuracy for .{} images is {}".format(test_class, acc))
        accuracy.append(acc)

    return accuracy


if __name__ == "__main__":
    start_time = time.time()
    yale_classes = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
    orl_classes = [1,2,3,4,5,6,7,8,9,10]

    ac_yale = execute('yale', yale_classes)
    sort = np.argsort(ac_yale)
    print ("The best image to classify is .{} and the worst is .{}\n".format(yale_classes[sort[len(yale_classes)-1]], yale_classes[sort[0]]))

    ac_orl = execute('orl', orl_classes)
    print ("Mean accuracy: {}\nStand deviation accuracy: {}\n".format(np.mean(ac_orl), np.std(ac_orl)))
    print ("The training and testing took {} seconds".format(time.time() - start_time))
