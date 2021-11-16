import nltk
import json
import random
import numpy
import tensorflow
import tflearn
import pickle
# nltk.download('punkt')

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, docs, training, output = pickle.load(f)

except:
    # seprate by tags and words
    ignore_letters = ['!', ',', '.', '?']
    lables = []
    words = []
    docs = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrd = nltk.word_tokenize(pattern)
            words.extend(wrd)
            docs.append((wrd, intent['tag']))
        lables.append(intent['tag'])

    words = [stemmer.stem(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))

    training = []
    output = []
    output_empty = [0] * len(lables)

    # convert raw data to training data
    for doc in docs:
        bag = []
        words_pattern = doc[0]
        words_pattern = [stemmer.stem(word.lower()) for word in words_pattern]
        for word in words:
            bag.append(1) if word in words_pattern else bag.append(0)

        output_row = list(output_empty)
        output_row[lables.index(doc[1])] = 1
        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, docs, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=500, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_word(s, words):
    bag = []
    new_words = nltk.word_tokenize(s)
    new_words = [stemmer.stem(word.lower()) for word in new_words]
    for w in words:
        bag.append(1) if w in new_words else bag.append(0)
    return numpy.array(bag)

def chatbot():
    print("Start talking with the BOT  !PRESS 'QUITE' TO EXIT \n")
    while True:
        inp = input("YOU: ")
        if inp.lower() == 'quite':
            break
        result = model.predict([bag_word(inp, words)])
        tag = lables[numpy.argmax(result)]

        for lab in data['intents']:
            if lab['tag'] == tag:
                responses = lab['responses']

        print(random.choice(responses))

if __name__ == "__main__":
    chatbot()