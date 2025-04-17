import numpy as np
import matplotlib.pyplot as plt
from src.gan_model import build_generator, train_gan
from src.data_preprocessing import load_and_preprocess_images
from src.style_transfer import apply_style_transfer

# Parameters
latent_dim = 100
img_shape = (128, 128, 3)
epochs = 5000  # Adjust based on compute power and dataset size

# Load and preprocess images from your training directory
data_dir = '../data/processed'
images = load_and_preprocess_images(data_dir, target_size=(128, 128))

# Build generator and discriminator (omitted here: create discriminator as well)
generator = build_generator(latent_dim, img_shape)

# Train GAN to generate base art
trained_generator = train_gan(generator, None, images, latent_dim, epochs=epochs)

# Generate a new piece of art
noise = np.random.normal(0, 1, (1, latent_dim))
generated_image = trained_generator.predict(noise)

# Optional: Apply a style transfer using a chosen style reference image
# For demonstration, the style_transfer function returns the original generated image.
stylized_image = apply_style_transfer(generated_image, None)

# Display the generated artwork
plt.imshow(stylized_image[0])
plt.axis('off')
plt.show()

# Optionally, save the image
import cv2
cv2.imwrite('../results/generated_art.jpg', cv2.cvtColor(stylized_image[0], cv2.COLOR_RGB2BGR))
print("Artwork generated and saved to results directory.")

