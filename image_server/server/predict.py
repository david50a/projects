from keras.models import Model
from keras.layers import Embedding, Input, LSTM, Dense, TimeDistributed, Concatenate, RepeatVector, Flatten
from keras.applications import ResNet50
import cv2
import numpy as np
def create_model(input_shape, vocab_size):
    embedding_size = 128
    max_len = 40

    image_input = Input(shape=input_shape)
    image_model = Dense(embedding_size, activation='relu')(image_input)
    image_model = RepeatVector(max_len)(image_model)

    language_input = Input(shape=(max_len,))
    language_model = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len)(language_input)
    language_model = LSTM(256, return_sequences=True)(language_model)
    language_model = TimeDistributed(Dense(embedding_size))(language_model)

    conca = Concatenate()([image_model, language_model])
    x = LSTM(128, return_sequences=True)(conca)
    x = LSTM(512, return_sequences=False)(x)
    output = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs=[image_input, language_input], outputs=output)
    return model

# Load vocabulary
vocab = np.load('vocab.npy', allow_pickle=True).item()
inv_vocab = {v: k for k, v in vocab.items()}
print("Vocabulary loaded")

# Update vocab_size to match the loaded vocabulary size
vocab_size = len(vocab) + 1

# Define model architecture
embedding_size = 128
max_len = 40

# Load ResNet model for feature extraction
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
image_input = Input(shape=(224, 224, 3))
image_features = resnet(image_input)
image_features = Flatten()(image_features)
image_features = Dense(embedding_size, activation='relu')(image_features)
image_features = RepeatVector(max_len)(image_features)

# Language model
language_input = Input(shape=(max_len,))
language_model = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len)(language_input)
language_model = LSTM(256, return_sequences=True)(language_model)
language_model = TimeDistributed(Dense(embedding_size))(language_model)

# Concatenate image and language features
concatenated_features = Concatenate()([image_features, language_model])

# LSTM layers
lstm_output = LSTM(128, return_sequences=True)(concatenated_features)
lstm_output = LSTM(512, return_sequences=False)(lstm_output)

# Output layer
output = Dense(vocab_size, activation='softmax')(lstm_output)

# Model
model = Model(inputs=[image_input, language_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
print("Model architecture defined")

# Load weights
input_shape = (2048,)  # Example input shape for ResNet50 features
model = create_model(input_shape, vocab_size)
model.load_weights('mine_model_weights.h5')
print("Weights loaded")

# Adjust input_shape according to your image feature shape
def generate_caption(image_path):
    # Image processing
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Extract image features using ResNet
    image_features = resnet.predict(image)  # Output shape will be (1, 7, 7, 2048)
    image_features = image_features.reshape(-1, 2048)  # Reshape to (49, 2048)

    # Generate caption
    start_token = np.zeros((49, max_len), dtype='int')  # Match the number of samples in image_features
    start_token[:, 0] = vocab['startofseq']
    caption = ''

    for i in range(1, max_len):
        # Predict next word given the image features and the previous token
        predictions = model.predict([image_features, start_token])

        # Sample the next word
        sampled_token_index = np.argmax(predictions[i, :])

        # Check if the end token is predicted
        if sampled_token_index == vocab['endofseq']:
            break

        # Append the sampled word to the caption
        sampled_word = inv_vocab[sampled_token_index]
        caption += ' ' + sampled_word

        # Update the start token for the next iteration
        start_token[:, i] = sampled_token_index

    return caption.strip()

#print(generate_caption(r"C:\Users\Owner\Downloads\imresizer-1715788416459.jpg"))