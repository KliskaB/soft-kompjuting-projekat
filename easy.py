import pytesseract
import cv2
import nltk
from nltk.tokenize import PunktSentenceTokenizer

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

image1 = cv2.imread("images/img.jpg")
string1 = pytesseract.image_to_string(image1)

image2 = cv2.imread("images/img2.jpg")
string2 = pytesseract.image_to_string(image2)

tokenizer = PunktSentenceTokenizer()
tokenized = tokenizer.tokenize(string2)


def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))


print(string2)
process_content()
