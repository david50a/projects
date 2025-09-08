import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import tensorflow_docs.vis.embed as embed
if not os.path.exists(r'training_checkout\ckpt'):
    os.mkdir(r'training_checkout\ckpt')
def make_generator_model():
    model = tf.keras.Sequential([
        # Fully connected layer, transforms the 100-dimensional input vector to 12544 (7*7*256)
        layers.Dense(12544, use_bias=False, input_shape=(100,)),
        # Batch normalization to stabilize and accelerate training
        layers.BatchNormalization(),
        # LeakyReLU activation: x if x > 0 else alpha * x (alpha ~0.2)
        layers.LeakyReLU(),
        # Reshape the vector into a 7x7x256 tensor (like a low-res image)
        layers.Reshape((7, 7, 256)),
        # Transposed convolution (upsampling) layer
        # Output size stays 7x7, 256 -> 128 channels
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        # Upsample to 14x14, reduce depth to 64
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        # Upsample to 28x28, output single channel (e.g. grayscale image)
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
    ])
    return model

def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        # Downsamples the input (28x28x1) to (14x14x64) using 64 filters of size 5x5 with stride 2
        layers.LeakyReLU(),  # Activation function that allows small gradients when input < 0
        layers.Dropout(0.3),  # Prevents overfitting by randomly dropping 30% of the neurons during training
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        # Further downsamples (14x14x64) to (7x7x128) using 128 filters of size 5x5 with stride 2
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        # Flattens the 3D tensor (7x7x128) into a 1D vector for the Dense layer
        layers.Dense(1)
        # Final output: a single scalar representing "real" or "fake" score (used with BinaryCrossentropy loss)
    ])
    return model

def discriminator_loss(real_output, fake_output):
    # Compare discriminator predictions on real images to a target tensor of ones (label: real)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # Compare discriminator predictions on fake images to a target tensor of zeros (label: fake)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # Total discriminator loss is the sum of both
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # Generator tries to fool the discriminator, so it wants the fake images to be classified as real (label: 1)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function # compiles the function into a TensorFlow graph for faster execution
def train_step(images):
    noise=tf.random.normal([NUMBER_OF_EXAPMLES_TO_GENERATE,NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:# open 2 gradient tapes for tracking the models
        generate_images=generator(noise,training=True)# generate a fake image and enables layers like BatchNorm and Dropout to behave properly during training
        real_output=discriminator(images,training=True)# evalute real image
        fake_output=discriminator(generate_images,training=True)# evaluate fake, generated image
        gen_loss=generator_loss(fake_output)# compute how well the generator fooled the disciminator
        disc_loss=discriminator_loss(real_output,fake_output)# compute the loss based on the ability of the distinguish real from fake images
    gradients_of_generator=gen_tape.gradient(gen_loss,generator.trainable_variables)# Calculates gradients of the generator loss with respect to generator weights (generator.trainable_variables returns the wieghts and biases)
    gradients_of_discriminator=disc_tape.gradient(disc_loss,discriminator.trainable_variables) # Calculates gradients of the dicriminator loss with respect to disciminator weights
    generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))# update the wieghts of the generator
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))# update the wieghts of the disciminator
(train_images,train_labels),(_,_)=tf.keras.datasets.mnist.load_data()
train_images=train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
train_images=(train_images-127.5)/127.5 #Normilaze the images to [-1,1]

def train(dataset,epochs):
    for epoch in range(epochs):
        start=time.time()
        for image_batch in dataset:
            train_step(image_batch)
        # produce images for GIF
        generate_and_save_images(generator,epoch+1,seed)# generate images using the current stage, give them names and display on maplotlib
        if epoch%15==14:
            checkpoint.save(file_prefix=checkout_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    # generate after the final epoch
    generate_and_save_images(generator,epochs,seed)

def generate_and_save_images(model,epoch,test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  fig=plt.figure(figsize=(4,4))# create a grid of 4*4 for 16 images
  for i in range(predictions.shape[0]):# Loops through each generated image in the batch 
      plt.subplot(4,4,i+1)# Puts each image in its grid position.
      plt.imshow(predictions[i,:,:,0]*127.5+127.5,cmap='gray')# Displays the image.  we use *127.5+127.5 because the generator use tanh activation
      plt.axis('off')
  plt.savefig('mnist_epochs\image_at_epoch{:04d}.png'.format(epoch))
  #plt.show()

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
if not os.path.exists('mnist_epochs'):
    os.mkdir('mnist_epochs')
BUFFER_SIZE=60000
BATCH_SIZE=256
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
generator=make_generator_model()
noise=tf.random.normal([1,100])# create a verctor with 100 random float values
generated_image=generator(noise,training=False)# this parameter will cintrol the behaver of the BatchNormalization layer (will use moving averages and no batch statistics)
plt.imshow(generated_image[0, :, :, 0], cmap='gray') # will show the first image with all the rows and columns in black white and set the colormap to grayscale
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)
# This method returns a helper function to compute cross entropy loss
cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)# lost function that use ofr binary classifications. use numbers and not propabilities and use sigmoid function to convret. used in case that the output layer is logits 
# Adam optimizer for the generator with a learning rate of 0.0001.
# This helps the generator update its weights during training.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# Adam optimizer for the discriminator with a learning rate of 0.0001.
# This helps the dicriminator update its weights during training.
discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
#create a TensorFlow checkpoint object that can save the state of the GAN
checkout_dir=r'training_checkout'
checkout_prefix=os.path.join(checkout_dir,'ckpt')
checkpoint=tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)

EPOCHS=50
NOISE_DIM=100
NUMBER_OF_EXAPMLES_TO_GENERATE=16
# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed=tf.random.normal([NUMBER_OF_EXAPMLES_TO_GENERATE,NOISE_DIM])
train(train_dataset,EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkout_dir))
display_image(EPOCHS)
anim_file = 'dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob(r'mnist_epochs\image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
embed.embed_file(anim_file)
