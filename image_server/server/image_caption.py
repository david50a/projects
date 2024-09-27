from keras.utils import to_categorical, plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed, Activation, \
    RepeatVector, Concatenate,Flatten,Convolution2D
import numpy as np
import cv2
from glob import glob
from keras.applications import ResNet50
from keras.models import Model, Sequential
import matplotlib.pyplot as plt

imges_path = r"C:\Users\Owner\Documents\school_project\dataset\archive\Images"
''''
images = glob(imges_path + '\*.jpg')
incept_model = ResNet50(include_top=True)
last = incept_model.layers[-2].output
model = Model(inputs=incept_model.input, outputs=last)
model.summary()
images_features = {}
count = 0
with open('predict.txt', 'w') as predict, open('image_name.txt', 'w') as image_name:
 for i in images:
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        pred = model.predict(img).reshape(2048, )
        img_name = i.split('/')[-1]
        images_features[img_name] = pred
        pred = str(pred)
        img_name = str(img_name)
        predict.write(pred + '@')
        image_name.write(img_name + "\n")
        count += 1
        if count > 4001:
            break
        elif count % 50 == 0:
            print(count)
'''''
incept_model = ResNet50(include_top=True)
last = incept_model.layers[-2].output
modele = Model(inputs = incept_model.input,outputs = last)
modele.summary()
images = open('image_name.txt', 'r').read().split('\n')
images_features = {}
with open('predict.txt', 'r') as prediction, open('image_name.txt', 'r') as name:
    image_names = name.read().split('\n')[:-1]
    predictions = prediction.read().split("@")
    print("iamge: ", len(image_names))
    print("caption: ", len(predictions))
    for i in range(len(image_names)):
        img = cv2.imread(image_names[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        predictions[i] = modele.predict(img).reshape(2048, )
    for i in range(len(image_names)):
        images_features[image_names[i].split('\\')[-1]] = predictions[i]
print(len(images_features))
caption_path = r"C:\Users\Owner\Documents\school_project\dataset\archive\captions.txt"
captions = open(caption_path, 'rb').read().decode('utf-8').split('\n')
print(len(captions))
captions_dict = {}
for i in captions:
    img_name = i.split(',')[0]
    caption = i.split(',')[1]
    if img_name in images_features:
        if img_name not in captions_dict:
            captions_dict[img_name] = [caption]
        else:
            captions_dict[img_name].append(caption)
print("caption: ", len(captions_dict))


def preprocessed(txt):
    modified = txt.lower()
    modified = 'startofseq ' + modified + ' endofseq'
    return modified


for k, v in captions_dict.items():
    for vv in v:
        captions_dict[k][v.index(vv)] = preprocessed(vv)
count_words = {}
count = 1
for k, vv in captions_dict.items():
    for v in vv:
        for word in v.split():
            if word not in count_words:
                count_words[word]=count
                count+=1

for k, vv in captions_dict.items():
    for v in vv:
        encoded = []
        for word in v.split():
            encoded.append(count_words[word])
        captions_dict[k][vv.index(v)] = encoded
print(len(captions_dict))
MAX_LEN = 0
for k, vv in captions_dict.items():
    for v in vv:
        if len(v) > MAX_LEN:
            MAX_LEN = len(v)
            print("v: ", len(v))

VOCAB_SIZE = len(count_words)
print("MAX_LEN: ",MAX_LEN)

def generator(photo, caption):
    X = []
    y_in = []
    y_out = []

    for k, vv in caption.items():
        for v in vv:
            for i in range(1, len(v)):
                X.append(photo[k])
                in_seq = [v[:i]]
                out_seq = v[i]
                in_seq = pad_sequences(in_seq, maxlen=MAX_LEN, padding='post', truncating='post')[0]
                out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE+1)[0]
                y_in.append(in_seq)
                y_out.append(out_seq)

    return X, y_in, y_out

x, y_in, y_out = generator(images_features, captions_dict)
x_in= np.array(x)
y_in = np.array(y_in, dtype='float64')
y_out = np.array(y_out, dtype='float64')
print(x_in.shape,y_in.shape,y_out.shape)
embedding_size = 128
max_len = MAX_LEN
vocab_size = len(count_words) + 1
image_model = Sequential()
image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

image_model.summary()

language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

language_model.summary()
conca = Concatenate()([image_model.output, language_model.output])
# Define LSTM layers
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs=out)
# model.load_weights(r'C:\Users\Owner\Documents\school_project\dataset\archive\model_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()
model.fit([x_in, y_in], y_out, batch_size=512, epochs=150)
inv_dict = {v: k for k, v in count_words.items()}
model.save('model.h5')
model.save_weights('mine_model_weights.h5')
np.save('vocab.npy', count_words)


def getImage(x):
    test_img_path = images[x]
    test_img = cv2.imread(test_img_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, (299, 299))
    test_img = np.reshape(test_img, (1, 299, 299, 3))
    return test_img


# test_feature=model.predict(getImage(200)).reshape(1,2048)
for i in range(5):
    no = np.random.randint(1500, 7000, (1, 1))[0, 0]
    test_feature = model.predict(getImage(no)).reshape(1, 2048)
    test_img_path = images[200]
    test_img = cv2.imread(test_img_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    plt.imshow(test_img)
    text_inp = ['startofseq']
    count = 0
    caption = ''
    while count < 20:
        count += 1
        encoded = []
        for i in text_inp:
            encoded.append(count_words[1])
            encoded = [encoded]
            encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)
            predction = np.argmax(model.predict([test_feature, encoded]))
            sampled_word = inv_dict[predction]
            caption = caption + ' ' + sampled_word
            if sampled_word == 'endofseq':
                break
            text_inp.append(sampled_word)
plt.figure()
plt.imshow(test_img)
plt.xlabel(caption)
