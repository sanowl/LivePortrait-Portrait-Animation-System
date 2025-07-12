

import os
import argparse
import logging
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import face_alignment
import fickling

# ===========================
# Configuration and Setup
# ===========================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Live Portrait Generation - Production Ready")
    
    # Training parameters
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for optimizers')
    parser.add_argument('--stage1_epochs', type=int, default=80, help='Number of epochs for Stage 1')
    parser.add_argument('--stage2_epochs', type=int, default=20, help='Number of epochs for Stage 2')
    
    # Paths
    parser.add_argument('--landmarks_cache', type=str, default='celeba_landmarks_cache.pkl', help='Path to landmarks cache file')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    
    # WandB
    parser.add_argument('--wandb_project', type=str, default='live-portrait-generation', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, required=True, help='WandB entity name')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

def setup_logging(save_dir):
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=[
                            logging.FileHandler(os.path.join(save_dir, 'training.log')),
                            logging.StreamHandler()
                        ])

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For Python random module if used
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ===========================
# Dataset Definition
# ===========================

class CelebADataset(Dataset):
    """
    CelebA Dataset with Landmarks

    This dataset loads images from the HuggingFace CelebA dataset and extracts facial landmarks.
    Landmarks are cached to disk to avoid redundant computations.
    """
    def __init__(self, split="train", transform=None, cache_file='celeba_landmarks_cache.pkl'):
        """
        Initialize the dataset.

        Args:
            split (str): Dataset split to load ('train', 'validation', etc.).
            transform (albumentations.Compose): Transformations to apply to images and landmarks.
            cache_file (str): Path to cache file for landmarks.
        """
        super().__init__()
        self.split = split
        self.transform = transform
        self.cache_file = cache_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dataset
        try:
            self.dataset = load_dataset("huggan/celeba-faces", split=split)
        except Exception as e:
            logging.error(f"Failed to load dataset split '{split}': {e}")
            raise e
        
        # Initialize face alignment
        try:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                                  flip_input=False, device=self.device)
        except Exception as e:
            logging.error(f"Failed to initialize face_alignment: {e}")
            raise e
        
        # Load or extract landmarks
        self.landmarks = self.load_or_extract_landmarks()
    
    def load_or_extract_landmarks(self):
        """
        Load landmarks from cache or extract them if cache doesn't exist.

        Returns:
            list: List of landmarks for each image.
        """
        if os.path.exists(self.cache_file):
            logging.info(f"Loading cached landmarks from {self.cache_file}...")
            try:
                with open(self.cache_file, "rb") as f:
                    landmarks = fickling.load(f)
                if len(landmarks) != len(self.dataset):
                    logging.warning("Cached landmarks size does not match dataset size. Re-extracting.")
                    return self.extract_landmarks()
                return landmarks
            except Exception as e:
                logging.error(f"Failed to load landmarks cache: {e}")
                logging.info("Re-extracting landmarks...")
                return self.extract_landmarks()
        else:
            logging.info("Landmarks cache not found. Extracting landmarks...")
            return self.extract_landmarks()
    
    def extract_landmarks(self):
        """
        Extract facial landmarks for all images in the dataset.

        Returns:
            list: List of landmarks for each image.
        """
        landmarks = []
        for idx, item in enumerate(tqdm(self.dataset, desc="Extracting Landmarks")):
            try:
                img = np.array(item['image'])
                detected_landmarks = self.fa.get_landmarks(img)
                if detected_landmarks and len(detected_landmarks) > 0:
                    landmarks.append(detected_landmarks[0])  # Assuming first face
                else:
                    logging.warning(f"No landmarks detected for image index {idx}. Using previous or default landmarks.")
                    if landmarks:
                        landmarks.append(landmarks[-1])  # Repeat last valid landmarks
                    else:
                        landmarks.append(np.zeros((68, 2)))  # Initialize with zeros
            except Exception as e:
                logging.error(f"Error extracting landmarks for image index {idx}: {e}")
                landmarks.append(np.zeros((68, 2)))  # Fallback
        
        # Save landmarks to cache
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(landmarks, f)
            logging.info(f"Landmarks cached to {self.cache_file}")
        except Exception as e:
            logging.error(f"Failed to cache landmarks: {e}")
        
        return landmarks
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx (int): Index of the data.

        Returns:
            tuple: (source_image, driving_image, source_landmarks, driving_landmarks)
        """
        try:
            source_item = self.dataset[idx]
            source_image = source_item["image"]
            source_landmarks = self.landmarks[idx]
            
            # Ensure driving image is different
            driving_idx = idx
            while driving_idx == idx:
                driving_idx = np.random.randint(len(self.dataset))
            driving_item = self.dataset[driving_idx]
            driving_image = driving_item["image"]
            driving_landmarks = self.landmarks[driving_idx]
            
            if self.transform:
                augmented_source = self.transform(image=np.array(source_image), keypoints=source_landmarks)
                augmented_driving = self.transform(image=np.array(driving_image), keypoints=driving_landmarks)
                
                # Convert keypoints to tensor
                source_kp = torch.tensor(augmented_source['keypoints'], dtype=torch.float32)
                driving_kp = torch.tensor(augmented_driving['keypoints'], dtype=torch.float32)
                
                return (augmented_source['image'], augmented_driving['image'], 
                        source_kp, driving_kp)
            else:
                return (source_image, driving_image, source_landmarks, driving_landmarks)
        except Exception as e:
            logging.error(f"Error fetching data at index {idx}: {e}")
            # Return dummy data or handle as needed
            return (torch.zeros(3, args.image_size, args.image_size), 
                    torch.zeros(3, args.image_size, args.image_size), 
                    torch.zeros(68, 2), torch.zeros(68, 2))

# ===========================
# Model Definitions
# ===========================

class AppearanceFeatureExtractor(nn.Module):
    """
    Extracts appearance features using a pre-trained ResNet50 model.
    """
    def __init__(self):
        super(AppearanceFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        return x

class MotionExtractor(nn.Module):
    """
    Extracts motion features from the driving image.
    """
    def __init__(self):
        super(MotionExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(512 * 8 * 8, 512)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class WarpingModule(nn.Module):
    """
    Warps appearance features based on motion features.
    """
    def __init__(self):
        super(WarpingModule, self).__init__()
        self.conv1 = nn.Conv2d(2048 + 512, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 2, kernel_size=3, padding=1)
    
    def forward(self, appearance_features, motion_features):
        b, c, h, w = appearance_features.shape
        motion_features = motion_features.view(b, 512, 1, 1).expand(-1, -1, h, w)
        combined = torch.cat([appearance_features, motion_features], dim=1)
        x = F.relu(self.conv1(combined))
        x = F.relu(self.conv2(x))
        flow = self.conv3(x)
        return flow

class Decoder(nn.Module):
    """
    Decoder network that generates the final image from features.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(1024),
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64)
        ])
    
    def forward(self, x):
        for layer, bn in zip(self.layers[:-1], self.batch_norms):
            x = F.relu(bn(layer(x)))
        x = torch.tanh(self.layers[-1](x))
        return x

class AttentionModule(nn.Module):
    """
    Attention mechanism to focus on relevant features.
    """
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x

class LivePortrait(nn.Module):
    """
    Live Portrait Generation Model combining feature extractors, warping, attention, and decoder.
    """
    def __init__(self):
        super(LivePortrait, self).__init__()
        self.appearance_extractor = AppearanceFeatureExtractor()
        self.motion_extractor = MotionExtractor()
        self.warping_module = WarpingModule()
        self.decoder = Decoder()
        self.attention = AttentionModule(2048)
    
    def forward(self, source_image, driving_image):
        appearance_features = self.appearance_extractor(source_image)
        motion_features = self.motion_extractor(driving_image)
        flow = self.warping_module(appearance_features, motion_features)
        warped_features = self.warp(appearance_features, flow)
        attended_features = self.attention(warped_features)
        generated_image = self.decoder(attended_features)
        return generated_image
    
    def warp(self, features, flow):
        """
        Apply spatial transformation to the features based on the flow.

        Args:
            features (torch.Tensor): Appearance features [B, C, H, W].
            flow (torch.Tensor): Flow vectors [B, 2, H, W].

        Returns:
            torch.Tensor: Warped features [B, C, H, W].
        """
        b, c, h, w = features.shape
        grid = self.get_normalized_grid(b, h, w).to(features.device)
        
        # Normalize flow to [-1, 1]
        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0, :, :] = (flow[:, 0, :, :] / (w - 1)) * 2
        flow_norm[:, 1, :, :] = (flow[:, 1, :, :] / (h - 1)) * 2
        final_grid = (grid + flow_norm).permute(0, 2, 3, 1)
        
        # Sample using grid_sample
        warped = F.grid_sample(features, final_grid, mode='bilinear', padding_mode='border')
        return warped
    
    @staticmethod
    def get_normalized_grid(b, h, w):
        """
        Create a normalized grid for grid_sample.

        Args:
            b (int): Batch size.
            h (int): Height of the grid.
            w (int): Width of the grid.

        Returns:
            torch.Tensor: Normalized grid [B, H, W, 2].
        """
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        grid = torch.stack((x, y), 2)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, H, W, 2]
        return grid

class Discriminator(nn.Module):
    """
    Discriminator network for adversarial training.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, H/2, W/2]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, H/4, W/4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, H/8, W/8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, H/16, W/16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # [B, 1, H/16-3, W/16-3]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss based on pre-trained VGG19 network.
    """
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4],    # Conv1_2
            vgg[4:9],   # Conv2_2
            vgg[9:18],  # Conv3_4
            vgg[18:27], # Conv4_4
            vgg[27:36]  # Conv5_4
        ])
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        # VGG normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
    
    def forward(self, x, y):
        """
        Compute perceptual loss between two images.

        Args:
            x (torch.Tensor): Generated image.
            y (torch.Tensor): Target image.

        Returns:
            torch.Tensor: Perceptual loss.
        """
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss

# ===========================
# Loss Functions
# ===========================

def calculate_adversarial_loss(output, discriminator):
    """
    Calculate adversarial loss for the generator.

    Args:
        output (torch.Tensor): Generated images.
        discriminator (nn.Module): Discriminator network.

    Returns:
        torch.Tensor: Adversarial loss.
    """
    fake_pred = discriminator(output)
    return F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

def calculate_real_adversarial_loss(target, discriminator):
    """
    Calculate adversarial loss for the discriminator on real images.

    Args:
        target (torch.Tensor): Real images.
        discriminator (nn.Module): Discriminator network.

    Returns:
        torch.Tensor: Real adversarial loss.
    """
    real_pred = discriminator(target)
    return F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))

def calculate_fake_adversarial_loss(output, discriminator):
    """
    Calculate adversarial loss for the discriminator on fake images.

    Args:
        output (torch.Tensor): Generated images.
        discriminator (nn.Module): Discriminator network.

    Returns:
        torch.Tensor: Fake adversarial loss.
    """
    fake_pred = discriminator(output.detach())
    return F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))

def calculate_stitching_loss(output, source, lambda_stitch=10.0):
    """
    Calculate stitching loss to preserve parts of the source image.

    Args:
        output (torch.Tensor): Generated images.
        source (torch.Tensor): Source images.
        lambda_stitch (float): Weight for stitching loss.

    Returns:
        torch.Tensor: Stitching loss.
    """
    mask = create_stitching_mask(source.shape).to(output.device)
    return F.l1_loss(output * mask, source * mask) * lambda_stitch

def create_stitching_mask(shape):
    """
    Create a stitching mask to preserve the left third of the image.

    Args:
        shape (tuple): Shape of the image tensor [B, C, H, W].

    Returns:
        torch.Tensor: Stitching mask.
    """
    mask = torch.ones(shape)
    mask[:, :, :, :shape[3]//3] = 0  # Zero out left third
    return mask

def calculate_eye_retargeting_loss(output, source_landmarks, driving_landmarks, lambda_eye=5.0):
    """
    Calculate loss to retarget eye movements.

    Args:
        output (torch.Tensor): Generated images.
        source_landmarks (torch.Tensor): Source landmarks.
        driving_landmarks (torch.Tensor): Driving landmarks.
        lambda_eye (float): Weight for eye retargeting loss.

    Returns:
        torch.Tensor: Eye retargeting loss.
    """
    source_eyes = extract_eye_landmarks(source_landmarks)
    driving_eyes = extract_eye_landmarks(driving_landmarks)
    output_eyes = extract_eye_region(output)
    target_eyes = driving_eyes - source_eyes + output_eyes
    return F.mse_loss(output_eyes, target_eyes) * lambda_eye

def extract_eye_landmarks(landmarks):
    """
    Extract eye landmarks from facial landmarks.

    Args:
        landmarks (torch.Tensor): Landmarks tensor [B, 68, 2].

    Returns:
        torch.Tensor: Concatenated left and right eye landmarks [B, 12].
    """
    # Assuming landmarks are in shape [B, 68, 2]
    left_eye = landmarks[:, 36:42].reshape(landmarks.size(0), -1)
    right_eye = landmarks[:, 42:48].reshape(landmarks.size(0), -1)
    return torch.cat([left_eye, right_eye], dim=1)  # [B, 12]

def extract_eye_region(image):
    """
    Extract the eye region from the image.

    Args:
        image (torch.Tensor): Image tensor [B, C, H, W].

    Returns:
        torch.Tensor: Eye region [B, C, H/3, W].
    """
    # Simple approximation: upper third of the image contains eyes
    return image[:, :, :image.shape[2]//3, :]  # [B, C, H/3, W]

def calculate_lip_retargeting_loss(output, source_landmarks, driving_landmarks, lambda_lip=5.0):
    """
    Calculate loss to retarget lip movements.

    Args:
        output (torch.Tensor): Generated images.
        source_landmarks (torch.Tensor): Source landmarks.
        driving_landmarks (torch.Tensor): Driving landmarks.
        lambda_lip (float): Weight for lip retargeting loss.

    Returns:
        torch.Tensor: Lip retargeting loss.
    """
    source_lips = extract_lip_landmarks(source_landmarks)
    driving_lips = extract_lip_landmarks(driving_landmarks)
    output_lips = extract_lip_region(output)
    target_lips = driving_lips - source_lips + output_lips
    return F.mse_loss(output_lips, target_lips) * lambda_lip

def extract_lip_landmarks(landmarks):
    """
    Extract lip landmarks from facial landmarks.

    Args:
        landmarks (torch.Tensor): Landmarks tensor [B, 68, 2].

    Returns:
        torch.Tensor: Lip landmarks [B, 40].
    """
    # Assuming landmarks are in shape [B, 68, 2]
    lips = landmarks[:, 48:].reshape(landmarks.size(0), -1)  # [B, 40]
    return lips

def extract_lip_region(image):
    """
    Extract the lip region from the image.

    Args:
        image (torch.Tensor): Image tensor [B, C, H, W].

    Returns:
        torch.Tensor: Lip region [B, C, H/3, W].
    """
    # Simple approximation: lower third of the image contains lips
    return image[:, :, 2*image.shape[2]//3:, :]  # [B, C, H/3, W]

def calculate_perceptual_loss(output, target, perceptual_loss_fn):
    """
    Calculate perceptual loss between output and target images.

    Args:
        output (torch.Tensor): Generated images.
        target (torch.Tensor): Target images.
        perceptual_loss_fn (nn.Module): Perceptual loss function.

    Returns:
        torch.Tensor: Perceptual loss.
    """
    return perceptual_loss_fn(output, target)

def calculate_total_loss_G(loss_adv, loss_perc, loss_stitch, loss_eye, loss_lip, 
                          stage=1, lambda_adv=1.0, lambda_perc=1.0, 
                          lambda_stitch=10.0, lambda_eye=5.0, lambda_lip=5.0):
    """
    Calculate the total generator loss based on the training stage.

    Args:
        loss_adv (torch.Tensor): Adversarial loss.
        loss_perc (torch.Tensor): Perceptual loss.
        loss_stitch (torch.Tensor): Stitching loss.
        loss_eye (torch.Tensor): Eye retargeting loss.
        loss_lip (torch.Tensor): Lip retargeting loss.
        stage (int): Current training stage.
        lambda_* (float): Weighting factors for each loss component.

    Returns:
        torch.Tensor: Total generator loss.
    """
    if stage == 1:
        return lambda_adv * loss_adv + lambda_perc * loss_perc
    else:
        return (lambda_adv * loss_adv + 
                lambda_perc * loss_perc + 
                lambda_stitch * loss_stitch + 
                lambda_eye * loss_eye + 
                lambda_lip * loss_lip)

# ===========================
# Training and Validation
# ===========================

def train_one_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, 
                   perceptual_loss_fn, stage, loss_weights):
    """
    Train the model for one epoch.

    Args:
        generator (nn.Module): Generator model.
        discriminator (nn.Module): Discriminator model.
        dataloader (DataLoader): Training data loader.
        optimizer_G (torch.optim.Optimizer): Optimizer for generator.
        optimizer_D (torch.optim.Optimizer): Optimizer for discriminator.
        device (torch.device): Computation device.
        perceptual_loss_fn (nn.Module): Perceptual loss function.
        stage (int): Current training stage.
        loss_weights (dict): Weighting factors for different loss components.

    Returns:
        tuple: (average generator loss, average discriminator loss)
    """
    generator.train()
    discriminator.train()
    running_loss_G, running_loss_D = 0.0, 0.0
    progress_bar = tqdm(dataloader, desc=f"Stage {stage} - Training")
    
    for batch_idx, (source_images, driving_images, source_landmarks, driving_landmarks) in enumerate(progress_bar):
        try:
            # Move data to device
            source_images = source_images.to(device)
            driving_images = driving_images.to(device)
            source_landmarks = source_landmarks.to(device)
            driving_landmarks = driving_landmarks.to(device)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            generated_images = generator(source_images, driving_images)
            
            loss_adv = calculate_adversarial_loss(generated_images, discriminator)
            loss_perc = calculate_perceptual_loss(generated_images, driving_images, perceptual_loss_fn)
            loss_stitch = calculate_stitching_loss(generated_images, source_images, lambda_stitch=loss_weights['lambda_stitch']) if stage == 2 else torch.tensor(0.0).to(device)
            loss_eye = calculate_eye_retargeting_loss(generated_images, source_landmarks, driving_landmarks, lambda_eye=loss_weights['lambda_eye']) if stage == 2 else torch.tensor(0.0).to(device)
            loss_lip = calculate_lip_retargeting_loss(generated_images, source_landmarks, driving_landmarks, lambda_lip=loss_weights['lambda_lip']) if stage == 2 else torch.tensor(0.0).to(device)
            
            loss_G = calculate_total_loss_G(loss_adv, loss_perc, loss_stitch, 
                                           loss_eye, loss_lip, stage=stage, 
                                           lambda_adv=loss_weights['lambda_adv'], 
                                           lambda_perc=loss_weights['lambda_perc'])
            
            loss_G.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real images
            loss_real = calculate_real_adversarial_loss(driving_images, discriminator)
            
            # Fake images
            loss_fake = calculate_fake_adversarial_loss(generated_images, discriminator)
            
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # Update running losses
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            
            # Update progress bar
            avg_G = running_loss_G / (batch_idx + 1)
            avg_D = running_loss_D / (batch_idx + 1)
            progress_bar.set_postfix({'G_Loss': f"{avg_G:.4f}", 'D_Loss': f"{avg_D:.4f}"})
        
        except Exception as e:
            logging.error(f"Error during training at batch {batch_idx}: {e}")
            continue  # Skip this batch
    
    epoch_loss_G = running_loss_G / len(dataloader)
    epoch_loss_D = running_loss_D / len(dataloader)
    return epoch_loss_G, epoch_loss_D

def validate(generator, dataloader, device, perceptual_loss_fn):
    """
    Validate the model on the validation set.

    Args:
        generator (nn.Module): Generator model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Computation device.
        perceptual_loss_fn (nn.Module): Perceptual loss function.

    Returns:
        float: Average validation loss.
    """
    generator.eval()
    val_loss = 0.0
    with torch.no_grad():
        for source_images, driving_images, _, _ in tqdm(dataloader, desc="Validating"):
            try:
                source_images = source_images.to(device)
                driving_images = driving_images.to(device)
                generated_images = generator(source_images, driving_images)
                loss = calculate_perceptual_loss(generated_images, driving_images, perceptual_loss_fn)
                val_loss += loss.item()
            except Exception as e:
                logging.error(f"Error during validation: {e}")
                continue  # Skip this batch
    average_val_loss = val_loss / len(dataloader)
    return average_val_loss

# ===========================
# Main Function
# ===========================

def main():
    """Main function to orchestrate training and validation."""
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(args.save_dir)
    logging.info("Starting Live Portrait Generation Training")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize WandB
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    # Define transformations
    transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    # Initialize dataset and dataloaders
    try:
        dataset = CelebADataset(split='train', transform=transform, cache_file=args.landmarks_cache)
        train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=args.seed)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                      num_workers=args.num_workers, pin_memory=args.pin_memory)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=args.pin_memory)
    except Exception as e:
        logging.error(f"Failed to initialize dataset or dataloaders: {e}")
        return
    
    # Initialize models
    try:
        generator = LivePortrait().to(device)
        discriminator = Discriminator().to(device)
        perceptual_loss_fn = VGGPerceptualLoss().to(device)
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        return
    
    # Initialize optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    # Initialize learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)
    
    # Watch models with WandB
    wandb.watch(generator, log="all")
    wandb.watch(discriminator, log="all")
    
    # Define loss weights
    loss_weights = {
        'lambda_adv': 1.0,
        'lambda_perc': 1.0,
        'lambda_stitch': 10.0,
        'lambda_eye': 5.0,
        'lambda_lip': 5.0
    }
    
    # Initialize best validation loss
    best_val_loss = float('inf')
    
    # Training Stages
    for stage, num_epochs in enumerate([args.stage1_epochs, args.stage2_epochs], start=1):
        logging.info(f"✨ Starting Stage {stage} training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            train_loss_G, train_loss_D = train_one_epoch(
                generator, discriminator, train_dataloader, 
                optimizer_G, optimizer_D, device, 
                perceptual_loss_fn, stage, loss_weights
            )
            val_loss = validate(generator, val_dataloader, device, perceptual_loss_fn)
            
            # Step schedulers
            scheduler_G.step()
            scheduler_D.step()
            
            # Log metrics
            wandb.log({
                "Stage": stage,
                "Epoch": epoch + (args.stage1_epochs if stage == 2 else 0),
                "Train_G_Loss": train_loss_G,
                "Train_D_Loss": train_loss_D,
                "Val_Perceptual_Loss": val_loss
            })
            
            # Print progress
            logging.info(f"Stage {stage} - Epoch [{epoch}/{num_epochs}], "
                         f"Train G_Loss: {train_loss_G:.4f}, Train D_Loss: {train_loss_D:.4f}, "
                         f"Val Perceptual Loss: {val_loss:.4f}")
            
            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path_G = os.path.join(args.save_dir, f"best_generator_stage{stage}.pth")
                checkpoint_path_D = os.path.join(args.save_dir, f"best_discriminator_stage{stage}.pth")
                torch.save(generator.state_dict(), checkpoint_path_G)
                torch.save(discriminator.state_dict(), checkpoint_path_D)
                logging.info(f"🔖 Saved new best models for Stage {stage} at epoch {epoch}")
        
        logging.info(f"🎉 Stage {stage} training completed.")
    
    # Final Model Saving
    final_gen_path = os.path.join(args.save_dir, "final_generator.pth")
    final_disc_path = os.path.join(args.save_dir, "final_discriminator.pth")
    try:
        torch.save(generator.state_dict(), final_gen_path)
        torch.save(discriminator.state_dict(), final_disc_path)
        logging.info(f"💾 Final models saved at {final_gen_path} and {final_disc_path}")
    except Exception as e:
        logging.error(f"Failed to save final models: {e}")
    
    # Finish WandB
    wandb.finish()
    logging.info("✅ Training process completed successfully.")

# ===========================
# Entry Point
# ===========================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Training terminated due to an unexpected error: {e}")
