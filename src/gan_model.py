import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim, img_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128 * 16 * 16, activation="relu", input_dim=latent_dim))
    model.add(layers.Reshape((16, 16, 128)))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(img_shape[2], kernel_size=3, padding="same", activation="tanh"))
    return model

def build_discriminator(img_shape):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def train_gan(generator, discriminator, images, latent_dim, epochs=10000, batch_size=64):
    # Prepare optimizers and loss function
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')
    z = tf.keras.Input(shape=(latent_dim,))
    img = generator(z)
    # For the combined model, freeze the discriminator's weights
    discriminator.trainable = False
    validity = discriminator(img)
    combined = tf.keras.Model(z, validity)
    combined.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')
    
    import numpy as np
    # Adversarial ground truths
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, images.shape[0], batch_size)
        real_imgs = images[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, real)
        
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
    return generator
