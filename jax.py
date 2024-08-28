import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from flax import linen as nn
from flax.training import train_state
import optax
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import face_alignment
import pickle
from typing import Dict, Any
from functools import partial

# Constants
IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE = 256, 32, 2e-4
STAGE_1_EPOCHS, STAGE_2_EPOCHS = 80, 20

# Check for GPU availability
device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
print(f"Using {device} device")

class CelebADataset:
    def __init__(self, split="train", transform=None):
        self.dataset = load_dataset("huggan/celeba-faces", split=split)
        self.transform = transform
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
        self.landmarks = self.extract_landmarks()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_image = self.dataset[idx]["image"]
        driving_idx = np.random.randint(len(self.dataset))
        driving_image = self.dataset[driving_idx]["image"]

        source_landmarks = self.landmarks[idx]
        driving_landmarks = self.landmarks[driving_idx]

        if self.transform:
            augmented_source = self.transform(image=np.array(source_image), keypoints=source_landmarks)
            augmented_driving = self.transform(image=np.array(driving_image), keypoints=driving_landmarks)
            return (jnp.array(augmented_source['image']), jnp.array(augmented_driving['image']), 
                    jnp.array(augmented_source['keypoints']), jnp.array(augmented_driving['keypoints']))
        return (jnp.array(source_image), jnp.array(driving_image), jnp.array(source_landmarks), jnp.array(driving_landmarks))

    def extract_landmarks(self):
        landmarks = []
        for item in tqdm(self.dataset, desc="Extracting landmarks"):
            img = np.array(item['image'])
            detected_landmarks = self.fa.get_landmarks(img)
            if detected_landmarks is not None:
                landmarks.append(detected_landmarks[0])
            else:
                landmarks.append(np.zeros((68, 2)))
        return landmarks

class AppearanceFeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        for features in [64, 128, 256, 512]:
            x = nn.Conv(features=features, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=False)(x)
            x = nn.relu(x)
        return nn.avg_pool(x, window_shape=(8, 8))

class MotionExtractor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        for features in [128, 256, 512]:
            x = nn.Conv(features=features, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(8, 8))
        return nn.Dense(features=512)(x.reshape((x.shape[0], -1)))

class WarpingModule(nn.Module):
    @nn.compact
    def __call__(self, appearance_features, motion_features):
        b, h, w, c = appearance_features.shape
        motion_features = jnp.broadcast_to(motion_features[:, :, None, None], (b, 512, h, w))
        combined = jnp.concatenate([appearance_features, motion_features], axis=1)
        x = nn.Conv(features=1024, kernel_size=(3, 3), padding="SAME")(combined)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        return nn.Conv(features=2, kernel_size=(3, 3), padding="SAME")(x)

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        for features in [1024, 512, 256, 128, 64]:
            x = nn.ConvTranspose(features=features, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=False)(x)
            x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        return nn.tanh(x)

class AttentionModule(nn.Module):
    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        query = nn.Conv(features=c // 8, kernel_size=(1, 1))(x)
        key = nn.Conv(features=c // 8, kernel_size=(1, 1))(x)
        value = nn.Conv(features=c, kernel_size=(1, 1))(x)

        query = query.reshape((b, h * w, c // 8))
        key = key.reshape((b, h * w, c // 8)).transpose((0, 2, 1))
        value = value.reshape((b, h * w, c))

        attention = jnp.matmul(query, key) / jnp.sqrt(c // 8)
        attention = nn.softmax(attention, axis=-1)

        out = jnp.matmul(attention, value)
        out = out.reshape((b, h, w, c))

        return self.param('gamma', nn.initializers.zeros, ()) * out + x

class LivePortrait(nn.Module):
    def setup(self):
        self.appearance_extractor = AppearanceFeatureExtractor()
        self.motion_extractor = MotionExtractor()
        self.warping_module = WarpingModule()
        self.decoder = Decoder()
        self.attention = AttentionModule()

    def __call__(self, source_image, driving_image):
        appearance_features = self.appearance_extractor(source_image)
        motion_features = self.motion_extractor(driving_image)
        flow = self.warping_module(appearance_features, motion_features)
        warped_features = self.warp(appearance_features, flow)
        attended_features = self.attention(warped_features)
        return self.decoder(attended_features)

    def warp(self, features, flow):
        b, h, w, c = features.shape
        grid = self.get_grid(b, h, w)
        final_grid = (grid + flow).transpose((0, 2, 3, 1))
        return jax.image.map_coordinates(features, final_grid, order=1, mode='nearest')

    @staticmethod
    def get_grid(b, h, w):
        x = jnp.linspace(-1, 1, w)
        y = jnp.linspace(-1, 1, h)
        xx, yy = jnp.meshgrid(x, y)
        return jnp.stack([xx, yy], axis=0)[None].repeat(b, axis=0)

class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x):
        for features in [64, 128, 256, 512]:
            x = nn.Conv(features=features, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            if features > 64:
                x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.Conv(features=1, kernel_size=(4, 4), strides=(1, 1), padding="VALID")(x)
        return nn.sigmoid(x)

class VGGPerceptualLoss(nn.Module):
    def setup(self):
        # Initialize VGG19 layers
        self.blocks = [
            self.vgg_block(64, 2),
            self.vgg_block(128, 2),
            self.vgg_block(256, 4),
            self.vgg_block(512, 4),
            self.vgg_block(512, 4)
        ]

    @nn.compact
    def __call__(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def vgg_block(self, features, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv(features=features, kernel_size=(3, 3), padding="SAME"))
            layers.append(nn.relu)
        layers.append(lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
        return nn.Sequential(layers)

def perceptual_loss(vgg, output, target):
    output_features = vgg(output)
    target_features = vgg(target)
    return sum(jnp.mean(jnp.abs(of - tf)) for of, tf in zip(output_features, target_features))

def adversarial_loss(discriminator_output):
    return -jnp.mean(jnp.log(discriminator_output + 1e-8))

def stitching_loss(output, source):
    mask = create_stitching_mask(source.shape)
    return jnp.mean(jnp.abs(output * mask - source * mask))

def create_stitching_mask(shape):
    mask = jnp.ones(shape)
    mask = mask.at[:, :, :shape[2]//3, :].set(0)
    return mask

def eye_retargeting_loss(output, source_landmarks, driving_landmarks):
    source_eyes = extract_eye_landmarks(source_landmarks)
    driving_eyes = extract_eye_landmarks(driving_landmarks)
    output_eyes = extract_eye_region(output)
    return jnp.mean((output_eyes - (driving_eyes - source_eyes + output_eyes))**2)

def extract_eye_landmarks(landmarks):
    left_eye = landmarks[:, 36:42]
    right_eye = landmarks[:, 42:48]
    return jnp.concatenate([left_eye, right_eye], axis=1)

def extract_eye_region(image):
    return image[:, :, :image.shape[2]//3, image.shape[3]//4:3*image.shape[3]//4]

def lip_retargeting_loss(output, source_landmarks, driving_landmarks):
    source_lips = extract_lip_landmarks(source_landmarks)
    driving_lips = extract_lip_landmarks(driving_landmarks)
    output_lips = extract_lip_region(output)
    return jnp.mean((output_lips - (driving_lips - source_lips + output_lips))**2)

def extract_lip_landmarks(landmarks):
    return landmarks[:, 48:]

def extract_lip_region(image):
    return image[:, :, 2*image.shape[2]//3:, image.shape[3]//4:3*image.shape[3]//4]

class TrainState(train_state.TrainState):
    discriminator: Any
    vgg: Any

@partial(jax.jit, static_argnums=(0,))
def train_step(apply_fn, state: TrainState, batch, rng):
    def loss_fn(params):
        source_images, driving_images, source_landmarks, driving_landmarks = batch
        generated_images = apply_fn({'params': params}, source_images, driving_images)
        
        discriminator_output = state.discriminator(generated_images)
        loss_adv = adversarial_loss(discriminator_output)
        loss_perceptual = perceptual_loss(state.vgg, generated_images, driving_images)
        loss_stitching = stitching_loss(generated_images, source_images)
        loss_eye = eye_retargeting_loss(generated_images, source_landmarks, driving_landmarks)
        loss_lip = lip_retargeting_loss(generated_images, source_landmarks, driving_landmarks)
        
        total_loss = loss_adv + loss_perceptual + loss_stitching + loss_eye + loss_lip
        return total_loss, (generated_images, loss_adv, loss_perceptual, loss_stitching, loss_eye, loss_lip)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (generated_images, loss_adv, loss_perceptual, loss_stitching, loss_eye, loss_lip)), grads = grad_fn(state.params)
    
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': loss,
        'loss_adv': loss_adv,
        'loss_perceptual': loss_perceptual,
        'loss_stitching': loss_stitching,
        'loss_eye': loss_eye,
        'loss_lip': loss_lip,
    }
    
    return new_state, metrics, generated_images

@jax.jit
def validate_step(apply_fn, params, vgg, batch):
    source_images, driving_images, _, _ = batch
    generated_images = apply_fn({'params': params}, source_images, driving_images)
    loss = perceptual_loss(vgg, generated_images, driving_images)
    return loss

def validate(apply_fn, state: TrainState, val_dataloader):
    val_loss = 0.0
    for batch in val_dataloader:
        val_loss += validate_step(apply_fn, state.params, state.vgg, batch)
    return val_loss / len(val_dataloader)

def create_train_state(rng, generator, discriminator, vgg, learning_rate):
    generator_params = generator.init(rng, jnp.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3)), jnp.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3)))
    discriminator_params = discriminator.init(rng, jnp.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3)))
    vgg_params = vgg.init(rng, jnp.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3)))

    tx = optax.adam(learning_rate, b1=0.5, b2=0.999)
    return TrainState.create(
        apply_fn=generator.apply,
        params=generator_params,
        tx=tx,
        discriminator=discriminator.apply({'params': discriminator_params}),
        vgg=vgg.apply({'params': vgg_params})
    )

def train(config):
    wandb.init(project="jax-live-portrait-generation", config=config)
    
    rng = random.PRNGKey(0)
    
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    dataset = CelebADataset(transform=transform)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    generator = LivePortrait()
    discriminator = Discriminator()
    vgg = VGGPerceptualLoss()

    rng, init_rng = random.split(rng)
    state = create_train_state(init_rng, generator, discriminator, vgg, LEARNING_RATE)

    best_val_loss = float('inf')

    print("✨ Starting Stage 1 training...")
    for epoch in range(STAGE_1_EPOCHS):
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{STAGE_1_EPOCHS}"):
            rng, step_rng = random.split(rng)
            state, metrics, generated_images = train_step(generator.apply, state, batch, step_rng)
            wandb.log(metrics)

        # Validation
        val_loss = validate(generator.apply, state, val_loader)
        print(f"Epoch [{epoch+1}/{STAGE_1_EPOCHS}], Val Loss: {val_loss:.4f}")
        wandb.log({"val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            with open("best_generator_stage1.pkl", "wb") as f:
                pickle.dump(state, f)

    print("🎉 Stage 1 training completed.")

    print("✨ Starting Stage 2 training...")
    for epoch in range(STAGE_2_EPOCHS):
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{STAGE_2_EPOCHS}"):
            rng, step_rng = random.split(rng)
            state, metrics, generated_images = train_step(generator.apply, state, batch, step_rng)
            wandb.log(metrics)

        # Validation
        val_loss = validate(generator.apply, state, val_loader)
        print(f"Epoch [{epoch+1}/{STAGE_2_EPOCHS}], Val Loss: {val_loss:.4f}")
        wandb.log({"val_loss": val_loss, "epoch": epoch + STAGE_1_EPOCHS})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            with open("best_generator_stage2.pkl", "wb") as f:
                pickle.dump(state, f)

    print("🎉 Stage 2 training completed.")

    # Final save
    with open("final_generator.pkl", "wb") as f:
        pickle.dump(state, f)
    print("💾 Final model saved successfully.")

    wandb.finish()

if __name__ == "__main__":
    config = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'stage_1_epochs': STAGE_1_EPOCHS,
        'stage_2_epochs': STAGE_2_EPOCHS,
        'image_size': IMAGE_SIZE,
    }
    train(config)