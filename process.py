import matplotlib
import numpy as np
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
import nltk
from keras import backend as K
import pickle
from ocr import CharacterRecognition
import matplotlib.pylab as plt
matplotlib.use('Agg')
from image_search import GoogleImagesDownloader


class POSTagger:

    def prepare_data1(self):
        tagged_sentences = nltk.corpus.treebank.tagged_sents()

        sentences, sentence_tags = [], []
        for tagged_sentence in tagged_sentences:
            sentence, tags = zip(*tagged_sentence)
            sentences.append(np.array(sentence))
            sentence_tags.append(np.array(tags))

        (train_sentences,
         test_sentences,
         train_tags,
         test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

        with open("serialization_folder/train_sentences.txt", "wb") as data_file1:
            pickle.dump(train_sentences, data_file1)

        with open("serialization_folder/test_sentences.txt", "wb") as data_file2:
            pickle.dump(test_sentences, data_file2)

        with open("serialization_folder/train_tags.txt", "wb") as data_file3:
            pickle.dump(train_tags, data_file3)

        with open("serialization_folder/test_tags.txt", "wb") as data_file4:
            pickle.dump(test_tags, data_file4)

        with open("serialization_folder/sentence_tags.txt", "wb") as data_file5:
            pickle.dump(sentence_tags, data_file5)

    def load_data(self):
        with open("serialization_folder/train_sentences.txt", "rb") as data_file1:
            train_sentences = pickle.load(data_file1)

        with open("serialization_folder/test_sentences.txt", "rb") as data_file2:
            test_sentences = pickle.load(data_file2)

        with open("serialization_folder/train_tags.txt", "rb") as data_file3:
            train_tags = pickle.load(data_file3)

        with open("serialization_folder/test_tags.txt", "rb") as data_file4:
            test_tags = pickle.load(data_file4)

        with open("serialization_folder/sentence_tags.txt", "rb") as data_file5:
            sentence_tags = pickle.load(data_file5)

        return train_sentences, train_tags, test_sentences, test_tags, sentence_tags

    def prepare_data2(self, train_sentences, train_tags, test_sentences, test_tags, sentence_tags):
        words, tags = set([]), set([])

        for ts in train_sentences:
            for w in ts:
                words.add(w.lower())

        for tt in sentence_tags:  # ovde je bilo train_tags, pa neke tagove ne prepoznaje!
            for t in tt:
                tags.add(t)

        word2index = {w: i + 2 for i, w in enumerate(list(words))}
        word2index['-PAD-'] = 0
        word2index['-OOV-'] = 1

        tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
        tag2index['-PAD-'] = 0

        train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

        for ts in train_sentences:
            s_int = []
            for w in ts:
                try:
                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])

            train_sentences_X.append(s_int)

        for ts in test_sentences:
            s_int = []
            for w in ts:
                try:
                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])

            test_sentences_X.append(s_int)

        for s in train_tags:
            train_tags_y.append([tag2index[t] for t in s])

        for s in test_tags:
            test_tags_y.append([tag2index[t] for t in s])

        MAX_LENGTH = len(max(train_sentences_X, key=len))

        train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
        test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
        train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
        test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

        return train_sentences_X, test_sentences_X, train_tags_y, test_tags_y, MAX_LENGTH, word2index, tag2index

    def to_categorical(self, sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)

    def ignore_accuracy(self, y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, 0), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    def create_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(MAX_LENGTH,)))
        model.add(Embedding(len(word2index), 128))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(TimeDistributed(Dense(len(tag2index))))
        model.add(Activation('softmax'))

        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy',
                      metrics=['accuracy', self.ignore_accuracy])

        return model

    def train_model(self, train_sentences_X, train_tags_y, tag2index):
        model.fit(train_sentences_X, self.to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=40,
                  validation_split=0.2)
        return model

    def save_model(self, model):
        model_structure = model.to_json()
        with open("serialization_folder/modelNew1.json", "w") as json_file:
            json_file.write(model_structure)
        model.save_weights("serialization_folder/modelNew1_weights.h5")
        print("Saved model to disk")

    def load_model(self):
        json_file = open('serialization_folder/modelNew1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model1 = model_from_json(loaded_model_json)
        model1.load_weights("serialization_folder/modelNew1_weights.h5")
        model1.summary()
        print("Model1 loaded.")
        return model1

    def logits_to_tokens(self, sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            token_sequence = []
            for categorical in categorical_sequence:
                token_sequence.append(index[np.argmax(categorical)])

            token_sequences.append(token_sequence)

        return token_sequences

    def logits_to_words(self, sequences, index):
        words_sequences = []
        for categorical_sequence in sequences:
            words_sequence = []
            for categorical in categorical_sequence:
                words_sequence.append(index[np.argmax(categorical)])

            words_sequences.append(words_sequence)

        return words_sequences

    def prepare_test_data(self, test_samples, word2index):
        test_samples_X = []
        for s in test_samples:
            s_int = []
            for w in s:
                try:
                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])
            test_samples_X.append(s_int)

        test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')
        return test_samples_X


tagger = POSTagger()
recognizer = CharacterRecognition()
downloader = GoogleImagesDownloader()

recognition_model = recognizer.load_trained_model()
alphabet = recognizer.form_alphabet()

tagger.prepare_data1()
train_sentences, train_tags, test_sentences, test_tags, sentence_tags = tagger.load_data()
train_sentences_X, test_sentences_X, train_tags_y, test_tags_y, MAX_LENGTH, word2index, tag2index \
    = tagger.prepare_data2(train_sentences, train_tags, test_sentences, test_tags, sentence_tags)

model = tagger.create_model()

model = tagger.train_model(train_sentences_X, train_tags_y, tag2index)

scores = model.evaluate(test_sentences_X, tagger.to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

tagger.save_model(model)

# model = tagger.load_model()

test_samples1 = [
    "running is very important for me .".split(),
    "I was running every day for a month .".split()
]

test_samples_X = tagger.prepare_test_data(test_samples1, word2index)

predictions = model.predict(test_samples_X)
print(predictions, predictions.shape)

print(tagger.logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))

img_color_test3 = recognizer.load_img('images/TEST.PNG')
img_test3 = recognizer.img_bin(recognizer.img_gray(img_color_test3))

selected_regions3, letters3, distances3 = recognizer.select_regions_training(img_color_test3.copy(), img_test3)
print('Number of regions:', len(letters3))
plt.imshow(selected_regions3)
plt.show()

distances3 = np.array(distances3).reshape(len(distances3), 1)
k_means3 = recognizer.k_meands_distances(distances3)

inputs3 = recognizer.prepare_for_model(letters3)
results3 = recognition_model.predict(np.array(inputs3, np.float32))
test_sample2 = recognizer.display_result(results3, alphabet, k_means3)
print(test_sample2)

test_sample2 = test_sample2.lower()
print(test_sample2.lower())
test_sample2 = test_sample2.split()
print(test_sample2)

test_samples_X2 = tagger.prepare_test_data(test_sample2, word2index)
predictions2 = model.predict(test_samples_X2)
print(predictions2, predictions2.shape)

print(test_samples_X2)
print(tagger.logits_to_tokens(predictions2, {i: t for t, i in tag2index.items()}))

token_sequence = tagger.logits_to_tokens(predictions2, {i: t for t, i in tag2index.items()})

duzina = len(test_sample2)
print(duzina)

def pairs(test_sample2, token_sequence):
    res = []
    for i in range(duzina):
        res.append([test_sample2[i], token_sequence[0][i]])
    return res

res = pairs(test_sample2, token_sequence)
print(res)

for word, tag in pairs(test_sample2, token_sequence):
    if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
        downloader.download_image(word)
        print("Downloaded images for ", word)
