#p9
# Experiment with neural networks like GANs (Generative Adversarial Networks) using Python libraries like TensorFlow or PyTorch to generate new images based on a dataset of images.

#pip install tensorflow matplotlib
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)

# Hyperparameters
BATCH_SIZE = 128
NOISE_DIM = 100
EPOCHS = 15

# Dataset pipeline
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=60000).batch(BATCH_SIZE)

# Generator model
generator = tf.keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(NOISE_DIM,)),
    layers.Dense(512, activation="relu"),
    layers.Dense(28 * 28, activation="sigmoid"),
    layers.Reshape((28, 28, 1))
])

# Discriminator model
discriminator = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(256, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile discriminator
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="binary_crossentropy"
)

# GAN model (generator + frozen discriminator)
discriminator.trainable = False
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="binary_crossentropy"
)

# Function to display generated images
def show_images():
    noise = tf.random.normal((16, NOISE_DIM))
    images = generator(noise, training=False)

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Training loop
for epoch in range(EPOCHS):
    for real_images in dataset:
        batch_size = real_images.shape[0]

        # Generate fake images
        noise = tf.random.normal((batch_size, NOISE_DIM))
        fake_images = generator(noise, training=True)

        # Train discriminator
        discriminator.trainable = True
        discriminator.train_on_batch(real_images, tf.ones((batch_size, 1)))
        discriminator.train_on_batch(fake_images, tf.zeros((batch_size, 1)))

        # Train generator via GAN
        noise = tf.random.normal((batch_size, NOISE_DIM))
        discriminator.trainable = False
        gan.train_on_batch(noise, tf.ones((batch_size, 1)))

    print(f"Epoch {epoch + 1}/{EPOCHS} completed")
    show_images()
