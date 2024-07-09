import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TOTAL_EPOCHS = 100
LATENT_DIM = 512

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set up distributed strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

def prepare_dataset(example):
    image = example['image']
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image

class AdaIN(keras.layers.Layer):
    def __init__(self, channels):
        super(AdaIN, self).__init__()
        self.channels = channels
        self.epsilon = 1e-5

    def call(self, content, style):
        content_mean, content_var = tf.nn.moments(content, axes=[1, 2], keepdims=True)
        style_mean, style_var = tf.nn.moments(style, axes=[1, 2], keepdims=True)
        normalized = (content - content_mean) / tf.sqrt(content_var + self.epsilon)
        return normalized * tf.sqrt(style_var + self.epsilon) + style_mean

class StyleEncoder(keras.Model):
    def __init__(self):
        super(StyleEncoder, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, 3, strides=2, padding='same')
        self.conv2 = keras.layers.Conv2D(128, 3, strides=2, padding='same')
        self.conv3 = keras.layers.Conv2D(256, 3, strides=2, padding='same')
        self.conv4 = keras.layers.Conv2D(512, 3, strides=2, padding='same')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(LATENT_DIM)

    def call(self, x):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.conv2(x))
        x = tf.nn.leaky_relu(self.conv3(x))
        x = tf.nn.leaky_relu(self.conv4(x))
        x = self.flatten(x)
        return self.dense(x)

class MappingNetwork(keras.Model):
    def __init__(self):
        super(MappingNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(512)
        self.dense2 = keras.layers.Dense(512)
        self.dense3 = keras.layers.Dense(512)
        self.dense4 = keras.layers.Dense(512)

    def call(self, x):
        x = tf.nn.leaky_relu(self.dense1(x))
        x = tf.nn.leaky_relu(self.dense2(x))
        x = tf.nn.leaky_relu(self.dense3(x))
        return self.dense4(x)

class StyleBasedGenerator(keras.Model):
    def __init__(self):
        super(StyleBasedGenerator, self).__init__()
        self.const_input = tf.Variable(tf.random.normal([1, 4, 4, 512]))
        self.conv1 = keras.layers.Conv2D(512, 3, padding='same')
        self.conv2 = keras.layers.Conv2D(512, 3, padding='same')
        self.conv3 = keras.layers.Conv2D(256, 3, padding='same')
        self.conv4 = keras.layers.Conv2D(128, 3, padding='same')
        self.conv5 = keras.layers.Conv2D(64, 3, padding='same')
        self.conv6 = keras.layers.Conv2D(3, 3, padding='same')
        self.adain1 = AdaIN(512)
        self.adain2 = AdaIN(512)
        self.adain3 = AdaIN(256)
        self.adain4 = AdaIN(128)
        self.adain5 = AdaIN(64)

    def call(self, w, motion_code, noise):
        x = tf.tile(self.const_input, [tf.shape(w)[0], 1, 1, 1])
        x = self.adain1(self.conv1(x), w)
        x = tf.nn.leaky_relu(x + noise[:, :4, :4, :])
        x = tf.image.resize(x, (8, 8))
        x = self.adain2(self.conv2(x), w)
        x = tf.nn.leaky_relu(x + noise[:, :8, :8, :])
        x = tf.image.resize(x, (16, 16))
        x = self.adain3(self.conv3(x), w)
        x = tf.nn.leaky_relu(x + noise[:, :16, :16, :])
        x = tf.image.resize(x, (32, 32))
        x = self.adain4(self.conv4(x), w)
        x = tf.nn.leaky_relu(x + noise[:, :32, :32, :])
        x = tf.image.resize(x, (64, 64))
        x = self.adain5(self.conv5(x), w)
        x = tf.nn.leaky_relu(x + noise[:, :64, :64, :])
        x = tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
        x = tf.tanh(self.conv6(x))
        return x

class MotionEncoder(keras.Model):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, 3, strides=2, padding='same')
        self.conv2 = keras.layers.Conv2D(128, 3, strides=2, padding='same')
        self.conv3 = keras.layers.Conv2D(256, 3, strides=2, padding='same')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(LATENT_DIM)

    def call(self, x):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.conv2(x))
        x = tf.nn.leaky_relu(self.conv3(x))
        x = self.flatten(x)
        return self.dense(x)

class AdaptiveWarpingModule(keras.Model):
    def __init__(self):
        super(AdaptiveWarpingModule, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, 3, padding='same')
        self.conv2 = keras.layers.Conv2D(128, 3, padding='same')
        self.conv3 = keras.layers.Conv2D(256, 3, padding='same')
        self.conv4 = keras.layers.Conv2D(2, 3, padding='same')

    def call(self, source, driving):
        x = tf.concat([source, driving], axis=-1)
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.conv2(x))
        x = tf.nn.leaky_relu(self.conv3(x))
        flow = self.conv4(x)
        return flow

class SelfAttentionModule(keras.layers.Layer):
    def __init__(self, channels):
        super(SelfAttentionModule, self).__init__()
        self.query = keras.layers.Conv2D(channels // 8, 1)
        self.key = keras.layers.Conv2D(channels // 8, 1)
        self.value = keras.layers.Conv2D(channels, 1)
        self.gamma = self.add_weight(name='gamma', shape=[], initializer='zeros')

    def call(self, x):
        batch_size, h, w, channels = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        query = tf.reshape(query, [batch_size, -1, h * w])
        key = tf.reshape(key, [batch_size, -1, h * w])
        value = tf.reshape(value, [batch_size, -1, h * w])
        
        attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True))
        out = tf.matmul(value, attention, transpose_b=True)
        out = tf.reshape(out, [batch_size, h, w, channels])
        
        return self.gamma * out + x

class AdvancedLivePortrait(keras.Model):
    def __init__(self):
        super(AdvancedLivePortrait, self).__init__()
        self.style_encoder = StyleEncoder()
        self.mapping_network = MappingNetwork()
        self.generator = StyleBasedGenerator()
        self.motion_encoder = MotionEncoder()
        self.warping_module = AdaptiveWarpingModule()
        self.attention = SelfAttentionModule(64)

    def call(self, source_image, driving_image, noise):
        style_code = self.style_encoder(source_image)
        motion_code = self.motion_encoder(driving_image)
        
        w = self.mapping_network(style_code)
        
        warping_params = self.warping_module(source_image, driving_image)
        warped_source = self.warp(source_image, warping_params)
        
        attention_output = self.attention(warped_source)
        
        output = self.generator(w, motion_code, noise)
        return output, attention_output

    def warp(self, image, flow):
        # Simple warping implementation
        grid = tf.meshgrid(tf.range(IMAGE_SIZE), tf.range(IMAGE_SIZE))
        grid = tf.stack(grid, axis=-1)
        grid = tf.cast(grid, tf.float32)
        
        flow = tf.cast(flow, tf.float32)
        sample_points = grid + flow
        
        warped = tf.image.sample_bilinear(image, sample_points)
        return warped

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, 4, strides=2, padding='same')
        self.conv2 = keras.layers.Conv2D(128, 4, strides=2, padding='same')
        self.conv3 = keras.layers.Conv2D(256, 4, strides=2, padding='same')
        self.conv4 = keras.layers.Conv2D(512, 4, strides=2, padding='same')
        self.conv5 = keras.layers.Conv2D(1, 4, strides=1, padding='valid')
        self.attention = SelfAttentionModule(256)

    def call(self, x):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.conv2(x))
        x = tf.nn.leaky_relu(self.conv3(x))
        x = self.attention(x)
        x = tf.nn.leaky_relu(self.conv4(x))
        x = self.conv5(x)
        return tf.squeeze(x, axis=[1, 2])

@tf.function
def train_step(source_images, driving_images, generator, discriminator, optimizer_G, optimizer_D):
    noise = tf.random.normal((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
    
    with tf.GradientTape(persistent=True) as tape:
        generated_images, attention_output = generator(source_images, driving_images, noise)
        
        real_output = discriminator(driving_images)
        fake_output = discriminator(generated_images)
        
        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)
        
        perceptual_loss = vgg_perceptual_loss(driving_images, generated_images)
        diversity_loss = perceptual_path_length(generator, source_images, noise)
        consistency_loss = temporal_consistency_loss(generator, source_images, driving_images)
        
        total_g_loss = g_loss + perceptual_loss + diversity_loss + consistency_loss
    
    gradients_G = tape.gradient(total_g_loss, generator.trainable_variables)
    gradients_D = tape.gradient(d_loss, discriminator.trainable_variables)
    
    optimizer_G.apply_gradients(zip(gradients_G, generator.trainable_variables))
    optimizer_D.apply_gradients(zip(gradients_D, discriminator.trainable_variables))
    
    return total_g_loss, d_loss

def generator_loss(fake_output):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_output, labels=tf.ones_like(fake_output)
    ))

def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=real_output, labels=tf.ones_like(real_output)
    ))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_output, labels=tf.zeros_like(fake_output)
    ))
    return real_loss + fake_loss

def vgg_perceptual_loss(real_images, generated_images):
    vgg = keras.applications.VGG19(include_top=False, weights='imagenet')
    real_features = vgg(real_images)
    generated_features = vgg(generated_images)
    return tf.reduce_mean(tf.abs(real_features - generated_features))

def perceptual_path_length(generator, source_images, noise):
    epsilon = 1e-4
    noise1 = noise
    noise2 = noise + epsilon * tf.random.normal(noise.shape)
    
    images1, _ = generator(source_images, source_images, noise1)
    images2, _ = generator(source_images, source_images, noise2)
    
    distance = tf.reduce_mean(tf.abs(images1 - images2)) / epsilon
    return distance

def temporal_consistency_loss(generator, source_images, driving_images):
    noise = tf.random.normal((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
    generated1, _ = generator(source_images, driving_images, noise)
    generated2, _ = generator(source_images, driving_images, noise)
    return tf.reduce_mean(tf.abs(generated1 - generated2))

def main():
    # Load and prepare dataset
    dataset = tfds.load('celeb_a', split='train')
    dataset = dataset.map(prepare_dataset)
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE)

    with strategy.scope():
        generator = AdvancedLivePortrait()
        discriminator = Discriminator()
        optimizer_G = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        optimizer_D = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(TOTAL_EPOCHS):
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCHS}")
        total_g_loss = 0
        total_d_loss = 0
        num_batches = 0
        
        for source_images, driving_images in dataset:
            g_loss, d_loss = train_step(source_images, driving_images, generator, discriminator, optimizer_G, optimizer_D)
            total_g_loss += g_loss
            total_d_loss += d_loss
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"  Batch {num_batches}: G_loss = {g_loss:.4f}, D_loss = {d_loss:.4f}")
        
        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Avg G_loss = {avg_g_loss:.4f}, Avg D_loss = {avg_d_loss:.4f}")
        
        # Save model checkpoints
        if (epoch + 1) % 10 == 0:
            generator.save_weights(f'generator_epoch_{epoch+1}.h5')
            discriminator.save_weights(f'discriminator_epoch_{epoch+1}.h5')
            print(f"Model checkpoints saved for epoch {epoch+1}")
        
        # Generate and save sample images
        if (epoch + 1) % 5 == 0:
            generate_and_save_samples(generator, dataset, epoch)

def generate_and_save_samples(generator, dataset, epoch):
    # Generate sample images
    num_samples = 4
    noise = tf.random.normal((num_samples, IMAGE_SIZE, IMAGE_SIZE, 1))
    source_images = next(iter(dataset))[0][:num_samples]
    driving_images = next(iter(dataset))[0][:num_samples]
    
    generated_images, _ = generator(source_images, driving_images, noise)
    
    # Combine images for visualization
    combined_images = tf.concat([source_images, driving_images, generated_images], axis=2)
    combined_images = (combined_images + 1) * 127.5
    combined_images = tf.clip_by_value(combined_images, 0, 255)
    combined_images = tf.cast(combined_images, tf.uint8)
    
    # Save the images
    tf.io.write_file(
        f'samples_epoch_{epoch+1}.png',
        tf.io.encode_png(tf.squeeze(combined_images))
    )
    print(f"Sample images saved for epoch {epoch+1}")

if __name__ == "__main__":
    main()