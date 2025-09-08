import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import tensorflow as tf
import tensorflow_docs.vis.embed as embed
from keras import layers
import imageio
import pandas as pd

def create_generator():
    model = tf.keras.Sequential([
        layers.Dense(7 * 6 * 512, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 6, 512)),

        layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),  # → (14, 12, 256)

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),  # → (28, 24, 128)

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),  # → (56, 48, 64)

        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),  # → (112, 96, 32)

        # 🔧 New layer added:
        layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),  # → (224, 192, 16)

        layers.Conv2D(3, (5, 5), padding='same', activation='tanh'),  # → (224, 192, 3)
        layers.Cropping2D(cropping=((3, 3), (7, 7)))  # → (218, 178, 3)
    ])
    return model

def create_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(218, 178, 3)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    noise = tf.random.normal([NUMBER_OF_EXAMPLES_TO_GENERATE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generate_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generate_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            images, _ = image_batch  # discard the labels
            train_step(images)

        generate_and_save_images(generator, epoch + 1, seed)

        if epoch % 15 == 14:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {:.2f} sec'.format(epoch + 1, time.time() - start))

    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2)  # Rescale from [-1,1] to [0,1]
        plt.axis('off')

    plt.savefig(f'CelebA_epochs/celeb_a_image_at_epoch{epoch:04d}.png')

def display_image(epoch_no):
    return PIL.Image.open(f'CelebA_epochs/celeb_a_image_at_epoch{epoch_no:04}.png')

# Setup directories
if not os.path.exists(r'CelebA_epochs'):
    os.makedirs('CelebA_epochs', exist_ok=True)
if not os.path.exists(r'Celeb_A_training_checkout/ckpt'):
    os.makedirs(r'Celeb_A_training_checkout/ckpt')

# Constants
IMAGE_DIR = r"CelebA\img_align_celeba\img_align_celeba"
ATTR_FILE = r"CelebA\list_attr_celeba.csv"
IMG_SIZE = (218, 178)
BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_DIM = 100
NUMBER_OF_EXAMPLES_TO_GENERATE = 16
EPOCHS = 50

# Load attributes
df_attr = pd.read_csv(ATTR_FILE)
df_attr.set_index("image_id", inplace=True)
df_attr = (df_attr + 1) // 2

# Load image paths and match labels
image_paths = []
image_labels = []

for image_id in df_attr.index:
    img_path = os.path.join(IMAGE_DIR, image_id)
    if os.path.exists(img_path):
        image_paths.append(img_path)
        image_labels.append(df_attr.loc[image_id].values.astype(np.float32))

# Preprocessing
def preprocess(img_path, label):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = (image / 127.5) - 1  # Rescale to [-1, 1]
    return image, label

# Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Models
generator = create_generator()
discriminator = create_discriminator()

# Optimizers and checkpoint
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = 'Celeb_A_training_checkout'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Generate and test one image before training
noise = tf.random.normal([1, NOISE_DIM])
generated_image = generator(noise, training=False)
decision = discriminator(generated_image)
plt.imshow((generated_image[0] + 1) / 2)  # display in [0,1] range
print("Discriminator decision:", decision)

# Train
seed = tf.random.normal([NUMBER_OF_EXAMPLES_TO_GENERATE, NOISE_DIM])
train(train_dataset, EPOCHS)

# Restore last checkpoint if needed
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display final image
display_image(EPOCHS)

# Create animation
anim_file = 'dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = sorted(glob.glob('CelebA_epochs/celeb_a_image_at_epoch*.png'))
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filenames[-1])
    writer.append_data(image)

embed.embed_file(anim_file)
