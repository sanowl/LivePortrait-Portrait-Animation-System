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
import os
import pickle

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
STAGE_1_EPOCHS = 80
STAGE_2_EPOCHS = 20
LANDMARKS_CACHE_FILE = "celeba_landmarks_cache.pkl"

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class CelebADataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.dataset = load_dataset("huggan/celeba-faces", split=split)
        self.transform = transform
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                              flip_input=False, device=device)
        self.landmarks = self.load_or_extract_landmarks()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_image = self.dataset[idx]["image"]
        # Ensure driving image is different
        while True:
            driving_idx = np.random.randint(len(self.dataset))
            if driving_idx != idx:
                break
        driving_image = self.dataset[driving_idx]["image"]
        
        source_landmarks = self.landmarks[idx]
        driving_landmarks = self.landmarks[driving_idx]
        
        if self.transform:
            augmented_source = self.transform(image=np.array(source_image), keypoints=source_landmarks)
            augmented_driving = self.transform(image=np.array(driving_image), keypoints=driving_landmarks)
            # Convert keypoints to tensor
            source_kp = torch.tensor(augmented_source['keypoints'], dtype=torch.float32)
            driving_kp = torch.tensor(augmented_driving['keypoints'], dtype=torch.float32)
            return (augmented_source['image'], augmented_driving['image'], 
                    source_kp, driving_kp)
        return (source_image, driving_image, source_landmarks, driving_landmarks)

    def load_or_extract_landmarks(self):
        if os.path.exists(LANDMARKS_CACHE_FILE):
            print("Loading cached landmarks...")
            with open(LANDMARKS_CACHE_FILE, "rb") as f:
                landmarks = pickle.load(f)
            return landmarks
        else:
            print("Extracting landmarks...")
            landmarks = []
            for item in tqdm(self.dataset, desc="Extracting landmarks"):
                img = np.array(item['image'])
                detected_landmarks = self.fa.get_landmarks(img)
                if detected_landmarks is not None and len(detected_landmarks) > 0:
                    landmarks.append(detected_landmarks[0])
                else:
                    # Handle missing landmarks by copying the previous or default
                    if landmarks:
                        landmarks.append(landmarks[-1])
                    else:
                        landmarks.append(np.zeros((68, 2)))
            # Cache landmarks to disk
            with open(LANDMARKS_CACHE_FILE, "wb") as f:
                pickle.dump(landmarks, f)
            return landmarks


class AppearanceFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
    
    def forward(self, x):
        x = self.features(x)
        return self.adaptive_pool(x)


class MotionExtractor(nn.Module):
    def __init__(self):
        super().__init__()
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
        return self.fc(x.view(x.size(0), -1))


class WarpingModule(nn.Module):
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        super().__init__()
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
        return torch.tanh(self.layers[-1](x))


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
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
    def __init__(self):
        super().__init__()
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
        return self.decoder(attended_features)
    
    def warp(self, features, flow):
        b, c, h, w = features.shape
        # Create normalized grid
        grid = self.get_grid(b, h, w).to(features.device)
        # Normalize flow to [-1, 1]
        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0, :, :] = (flow[:, 0, :, :] / (w - 1)) * 2
        flow_norm[:, 1, :, :] = (flow[:, 1, :, :] / (h - 1)) * 2
        final_grid = (grid + flow_norm).permute(0, 2, 3, 1)
        return F.grid_sample(features, final_grid, mode='bilinear', padding_mode='border')
    
    @staticmethod
    def get_grid(b, h, w):
        # Create a mesh grid in the range [-1, 1]
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        grid = torch.stack((x, y), 2)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, H, W, 2]
        return grid


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4],    # Conv1_2
            vgg[4:9],   # Conv2_2
            vgg[9:18],  # Conv3_4
            vgg[18:27], # Conv4_4
            vgg[27:36]  # Conv5_4
        ])
        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
    
    def forward(self, x, y):
        # Normalize inputs as per VGG requirements
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss


def calculate_perceptual_loss(output, target, perceptual_loss_fn):
    return perceptual_loss_fn(output, target)


def calculate_adversarial_loss(output, discriminator):
    fake_pred = discriminator(output)
    return F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))


def calculate_real_adversarial_loss(target, discriminator):
    real_pred = discriminator(target)
    return F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))


def calculate_fake_adversarial_loss(output, discriminator):
    fake_pred = discriminator(output.detach())
    return F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))


def calculate_stitching_loss(output, source, lambda_stitch=10.0):
    mask = create_stitching_mask(source.shape).to(output.device)
    return F.l1_loss(output * mask, source * mask) * lambda_stitch


def create_stitching_mask(shape):
    mask = torch.ones(shape)
    # Zero out the left third of the image as an example
    mask[:, :, :, :shape[3]//3] = 0
    return mask


def calculate_eye_retargeting_loss(output, source_landmarks, driving_landmarks, lambda_eye=5.0):
    source_eyes = extract_eye_landmarks(source_landmarks)
    driving_eyes = extract_eye_landmarks(driving_landmarks)
    output_eyes = extract_eye_region(output)
    target_eyes = driving_eyes - source_eyes + output_eyes
    return F.mse_loss(output_eyes, target_eyes) * lambda_eye


def extract_eye_landmarks(landmarks):
    # landmarks: [B, 68, 2]
    left_eye = landmarks[:, 36:42].reshape(landmarks.size(0), -1)
    right_eye = landmarks[:, 42:48].reshape(landmarks.size(0), -1)
    return torch.cat([left_eye, right_eye], dim=1)  # [B, 12]

def extract_eye_region(image):
    # Simple approximation: upper third of the face
    return image[:, :, :image.shape[2]//3, :]  # [B, C, H/3, W]

def calculate_lip_retargeting_loss(output, source_landmarks, driving_landmarks, lambda_lip=5.0):
    source_lips = extract_lip_landmarks(source_landmarks)
    driving_lips = extract_lip_landmarks(driving_landmarks)
    output_lips = extract_lip_region(output)
    target_lips = driving_lips - source_lips + output_lips
    return F.mse_loss(output_lips, target_lips) * lambda_lip


def extract_lip_landmarks(landmarks):
    # landmarks: [B, 68, 2]
    return landmarks[:, 48:].reshape(landmarks.size(0), -1)  # [B, 40]

def extract_lip_region(image):
    # Simple approximation: lower third of the face
    return image[:, :, 2*image.shape[2]//3:, :]  # [B, C, H/3, W]


def calculate_total_loss_G(g_loss_adv, g_loss_perc, g_loss_stitch, g_loss_eye, g_loss_lip, 
                          stage=1, lambda_adv=1.0, lambda_perc=1.0, lambda_stitch=10.0, 
                          lambda_eye=5.0, lambda_lip=5.0):
    if stage == 1:
        return lambda_adv * g_loss_adv + lambda_perc * g_loss_perc
    else:
        return (lambda_adv * g_loss_adv + 
                lambda_perc * g_loss_perc + 
                lambda_stitch * g_loss_stitch + 
                lambda_eye * g_loss_eye + 
                lambda_lip * g_loss_lip)


def train_one_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, 
                   perceptual_loss_fn, stage):
    generator.train()
    discriminator.train()
    running_loss_G, running_loss_D = 0.0, 0.0
    progress_bar = tqdm(dataloader, desc=f"Stage {stage} - Training")
    for batch_idx, (source_images, driving_images, source_landmarks, driving_landmarks) in enumerate(progress_bar):
        source_images = source_images.to(device)
        driving_images = driving_images.to(device)
        source_landmarks = source_landmarks.to(device)
        driving_landmarks = driving_landmarks.to(device)
        
        batch_size = source_images.size(0)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()
        generated_images = generator(source_images, driving_images)
        
        loss_adv = calculate_adversarial_loss(generated_images, discriminator)
        loss_perceptual = calculate_perceptual_loss(generated_images, driving_images, perceptual_loss_fn)
        loss_stitching = calculate_stitching_loss(generated_images, source_images) if stage == 2 else 0.0
        loss_eye = calculate_eye_retargeting_loss(generated_images, source_landmarks, driving_landmarks) if stage == 2 else 0.0
        loss_lip = calculate_lip_retargeting_loss(generated_images, source_landmarks, driving_landmarks) if stage == 2 else 0.0
        
        loss_G = calculate_total_loss_G(loss_adv, loss_perceptual, loss_stitching, 
                                       loss_eye, loss_lip, stage)
        
        loss_G.backward()
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real images
        real_pred = discriminator(driving_images)
        loss_real = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
        
        # Fake images
        fake_pred = discriminator(generated_images.detach())
        loss_fake = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
        
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        
        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()
        
        avg_G = running_loss_G / (batch_idx + 1)
        avg_D = running_loss_D / (batch_idx + 1)
        progress_bar.set_postfix({'G_Loss': f"{avg_G:.4f}", 'D_Loss': f"{avg_D:.4f}"})
    
    epoch_loss_G = running_loss_G / len(dataloader)
    epoch_loss_D = running_loss_D / len(dataloader)
    return epoch_loss_G, epoch_loss_D


def validate(generator, dataloader, device, perceptual_loss_fn):
    generator.eval()
    val_loss = 0.0
    with torch.no_grad():
        for source_images, driving_images, _, _ in tqdm(dataloader, desc="Validating"):
            source_images = source_images.to(device)
            driving_images = driving_images.to(device)
            generated_images = generator(source_images, driving_images)
            loss = calculate_perceptual_loss(generated_images, driving_images, perceptual_loss_fn)
            val_loss += loss.item()
    return val_loss / len(dataloader)


def main():
    wandb.init(project="live-portrait-generation", entity="your-entity-name", config={
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "stage_1_epochs": STAGE_1_EPOCHS,
        "stage_2_epochs": STAGE_2_EPOCHS,
    })
    
    print(f"Using device: {device}")
    
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    dataset = CelebADataset(transform=transform)
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    generator = LivePortrait().to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)
    
    wandb.watch(generator, log="all")
    wandb.watch(discriminator, log="all")
    
    best_val_loss = float('inf')
    
    # Stage 1 Training
    print("✨ Starting Stage 1 training...")
    for epoch in range(1, STAGE_1_EPOCHS + 1):
        train_loss_G, train_loss_D = train_one_epoch(
            generator, discriminator, train_dataloader, 
            optimizer_G, optimizer_D, device, 
            perceptual_loss_fn, stage=1
        )
        val_loss = validate(generator, val_dataloader, device, perceptual_loss_fn)
        
        scheduler_G.step()
        scheduler_D.step()
        
        print(f"Stage 1 - Epoch [{epoch}/{STAGE_1_EPOCHS}], "
              f"Train G_Loss: {train_loss_G:.4f}, Train D_Loss: {train_loss_D:.4f}, Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "Stage": 1,
            "Train_G_Loss": train_loss_G,
            "Train_D_Loss": train_loss_D,
            "Val_Loss": val_loss,
            "Epoch": epoch
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), "best_generator_stage1.pth")
            torch.save(discriminator.state_dict(), "best_discriminator_stage1.pth")
            print("🔖 Saved new best model for Stage 1.")
    
    print("🎉 Stage 1 training completed.")
    
    # Stage 2 Training
    print("✨ Starting Stage 2 training...")
    for epoch in range(1, STAGE_2_EPOCHS + 1):
        train_loss_G, train_loss_D = train_one_epoch(
            generator, discriminator, train_dataloader, 
            optimizer_G, optimizer_D, device, 
            perceptual_loss_fn, stage=2
        )
        val_loss = validate(generator, val_dataloader, device, perceptual_loss_fn)
        
        scheduler_G.step()
        scheduler_D.step()
        
        total_epoch = STAGE_1_EPOCHS + epoch
        print(f"Stage 2 - Epoch [{epoch}/{STAGE_2_EPOCHS}], "
              f"Train G_Loss: {train_loss_G:.4f}, Train D_Loss: {train_loss_D:.4f}, Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "Stage": 2,
            "Train_G_Loss": train_loss_G,
            "Train_D_Loss": train_loss_D,
            "Val_Loss": val_loss,
            "Epoch": total_epoch
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), "best_generator_stage2.pth")
            torch.save(discriminator.state_dict(), "best_discriminator_stage2.pth")
            print("🔖 Saved new best model for Stage 2.")
    
    print("🎉 Stage 2 training completed.")
    
    # Final save
    torch.save(generator.state_dict(), "final_generator.pth")
    torch.save(discriminator.state_dict(), "final_discriminator.pth")
    print("💾 Final models saved successfully.")
    
    wandb.finish()


if __name__ == "__main__":
    main()
