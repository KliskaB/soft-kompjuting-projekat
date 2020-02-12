from __future__ import print_function
import cv2
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD, Adam
from keras.models import model_from_json
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pylab as plt


class CharacterRecognition:

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def img_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def img_bin(self, image_gs):
        ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
        return image_bin

    def resize_region(self, region):
        resized = cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)
        return resized

    def scale_to_range(self, image):
        return image / 255

    def matrix_to_vector(self, image):
        return image.flatten()

    def prepare_for_model(self, regions):
        ready_for_model = []
        for region in regions:
            ready_for_model.append(self.matrix_to_vector(self.scale_to_range(region)))
        return ready_for_model

    def convert_output(self, outputs):
        return np.eye(len(outputs))

    def winner(self, output):
        return max(enumerate(output), key=lambda x: x[1])[0]

    def select_regions_training(self, image_orig, image_bin):
        contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        regions_array = []

        i = 0
        for contour in contours:
            i = i + 1
            x, y, w, h = cv2.boundingRect(contour)
            region = image_bin[y:y + h + 1, x:x + w + 1]

            area = h * w

            if 400 < area < 1500 or 31 > h > 32 or 23 <= h < 50:
                if i == 33 or i == 36 or i == 37:
                    continue
                regions_array.append([self.resize_region(region), (x, y, w, h)])
                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 1)
                regions_array = sorted(regions_array, key=lambda item: item[1][0])

                sorted_regions = [region[0] for region in regions_array]
                sorted_rectangles = [region[1] for region in regions_array]
                # print("%d." % i, "x = %f" % x, "w = %f" % w, "y = %f" % y, "h = %f" % h)

                region_distances = []
                for index in range(0, len(sorted_rectangles) - 1):
                    current = sorted_rectangles[index]
                    next_rect = sorted_rectangles[index + 1]
                    distance = next_rect[0] - (current[0] + current[2])
                    region_distances.append(distance)

        return image_orig, sorted_regions, region_distances

    def select_regions_testing(self, image_orig, image_bin):
        contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        regions_array = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            region = image_bin[y:y + h + 1, x:x + w + 1]

            regions_array.append([self.resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 1)
            regions_array = sorted(regions_array, key=lambda item: item[1][0])

            sorted_regions = [region[0] for region in regions_array]
            sorted_rectangles = [region[1] for region in regions_array]

            region_distances = []
            for index in range(0, len(sorted_rectangles) - 1):
                current = sorted_rectangles[index]
                next_rect = sorted_rectangles[index + 1]
                distance = next_rect[0] - (current[0] + current[2])
                region_distances.append(distance)

        return image_orig, sorted_regions, region_distances

    def form_alphabet(self):
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U',
                    'V', 'W', 'X', 'Y', 'Z',
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u',
                    'v', 'w', 'x', 'y', 'z']
        return alphabet

    def create_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=784, activation='sigmoid'))
        model.add(Dense(52, activation='sigmoid'))
        return model

    def train_model(self, model, X_train, y_train):
        X_train = np.array(X_train, np.float32)
        y_train = np.array(y_train, np.float32)

        sgd = SGD(lr=0.01, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer=sgd)

        model.fit(X_train, y_train, epochs=3000, batch_size=1, verbose=0, shuffle=False)
        return model


    def create_model1(self):
        model = Sequential()
        model.add(Dense(128, input_dim=784, activation='relu'))
        model.add(Dense(52, activation='relu'))
        return model

    def train_model1(self, model, X_train, y_train):
        X_train = np.array(X_train, np.float32)
        y_train = np.array(y_train, np.float32)

        model.compile(loss='categorical_crossentropy', optimizer=Adam())

        model.fit(X_train, y_train, epochs=3000, batch_size=1, verbose=0, shuffle=False)
        return model

    def save_model(self, model):
        model_json = model.to_json()
        with open("serialization_folder/prepoznavanjeNeuronska3.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("serialization_folder/prepoznavanjeNeuronska3.h5")

    def load_trained_model(self):
        try:
            json_file = open('serialization_folder/prepoznavanjeNeuronska3.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights("serialization_folder/prepoznavanjeNeuronska3.h5")
            print("Trained model successfully saved.")
            return model
        except Exception as e:
            return None

    def display_result(self, outputs, alphabet, k_means):
        w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
        result = alphabet[self.winner(outputs[0])]
        for idx, output in enumerate(outputs[1:, :]):
            if (k_means.labels_[idx] == w_space_group):
                result += ' '
            result += alphabet[self.winner(output)]
        return result

    def k_meands_distances(self, distances):
        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        k_means.fit(distances)
        return k_means


recognizer = CharacterRecognition()

img_color = recognizer.load_img('images/alfabet.PNG')
img = recognizer.img_bin(recognizer.img_gray(img_color))
selected_regions, letters, region_distances = recognizer.select_regions_training(img_color.copy(), img)
print('Number of regions:', len(letters))
# plt.imshow(selected_regions)
# plt.show()

alphabet = recognizer.form_alphabet()

inputs = recognizer.prepare_for_model(letters)
outputs = recognizer.convert_output(alphabet)

model = recognizer.load_trained_model()

if model == None:
    print("Training started.")
    model = recognizer.create_model1()
    model = recognizer.train_model1(model, inputs, outputs)
    print("Traning is over.")
    recognizer.save_model(model)

img_color_test1 = recognizer.load_img('images/test6.PNG')
img_color_test2 = recognizer.load_img('images/test4.PNG')
img_color_test3 = recognizer.load_img('images/TEST.PNG')

img_test1 = recognizer.img_bin(recognizer.img_gray(img_color_test1))
img_test2 = recognizer.img_bin(recognizer.img_gray(img_color_test2))
img_test3 = recognizer.img_bin(recognizer.img_gray(img_color_test3))

selected_regions1, letters1, distances1 = recognizer.select_regions_training(img_color_test1.copy(), img_test1)
print('Number of regions:', len(letters1))
# plt.imshow(selected_regions1)
# plt.show()

selected_regions2, letters2, distances2 = recognizer.select_regions_training(img_color_test2.copy(), img_test2)
print('Number of regions:', len(letters2))
# plt.imshow(selected_regions2)
# plt.show()

selected_regions3, letters3, distances3 = recognizer.select_regions_training(img_color_test3.copy(), img_test3)
print('Number of regions:', len(letters3))
plt.imshow(selected_regions3)
plt.show()

distances1 = np.array(distances1).reshape(len(distances1), 1)
k_means1 = recognizer.k_meands_distances(distances1)

distances2 = np.array(distances2).reshape(len(distances2), 1)
k_means2 = recognizer.k_meands_distances(distances2)

distances3 = np.array(distances3).reshape(len(distances3), 1)
k_means3 = recognizer.k_meands_distances(distances3)

inputs1 = recognizer.prepare_for_model(letters1)
results1 = model.predict(np.array(inputs1, np.float32))
print(recognizer.display_result(results1, alphabet, k_means1))

inputs2 = recognizer.prepare_for_model(letters2)
results2 = model.predict(np.array(inputs2, np.float32))
print(recognizer.display_result(results2, alphabet, k_means2))

inputs3 = recognizer.prepare_for_model(letters3)
results3 = model.predict(np.array(inputs3, np.float32))
print(recognizer.display_result(results3, alphabet, k_means3))
