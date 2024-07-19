import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import face_alignment

IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE = 256, 32, 2e-4
STAGE_1_EPOCHS, STAGE_2_EPOCHS = 80, 20

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class CelebADataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.dataset = load_dataset("huggan/celeba-faces", split=split)
        self.transform = transform
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
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
            return (augmented_source['image'], augmented_driving['image'], 
                    augmented_source['keypoints'], augmented_driving['keypoints'])
        return (source_image, driving_image, source_landmarks, driving_landmarks)

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
        return self.conv3(x)

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
        grid = self.get_grid(b, h, w).to(features.device)
        final_grid = (grid + flow).permute(0, 2, 3, 1)
        return F.grid_sample(features, final_grid, mode='bilinear', padding_mode='border')
    
    @staticmethod
    def get_grid(b, h, w):
        xx, yy = torch.meshgrid(torch.arange(w), torch.arange(h))
        return torch.stack((xx.repeat(b, 1, 1), yy.repeat(b, 1, 1)), 1).float()

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
            vgg[:4],
            vgg[4:9],
            vgg[9:18],
            vgg[18:27],
            vgg[27:36]
        ])
        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False
    
    def forward(self, x, y):
        loss = 0.0
        for block in self.blocks:
            x, y = block(x), block(y)
            loss += F.l1_loss(x, y)
        return loss

def calculate_perceptual_loss(output, target, perceptual_loss):
    return perceptual_loss(output, target)

def calculate_adversarial_loss(output, discriminator):
    fake_pred = discriminator(output)
    return F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

def calculate_stitching_loss(output, source):
    mask = create_stitching_mask(source.shape).to(output.device)
    return F.l1_loss(output * mask, source * mask)

def create_stitching_mask(shape):
    mask = torch.ones(shape)
    mask[:, :, :shape[2]//3, :] = 0 
    return mask

def calculate_eye_retargeting_loss(output, source_landmarks, driving_landmarks):
    source_eyes = extract_eye_landmarks(source_landmarks)
    driving_eyes = extract_eye_landmarks(driving_landmarks)
    output_eyes = extract_eye_region(output)
    return F.mse_loss(output_eyes, driving_eyes - source_eyes + output_eyes)

def extract_eye_landmarks(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    return torch.cat([left_eye, right_eye])

def extract_eye_region(image):
    return image[:, :, :image.shape[2]//3, image.shape[3]//4:3*image.shape[3]//4]

def calculate_lip_retargeting_loss(output, source_landmarks, driving_landmarks):
    source_lips = extract_lip_landmarks(source_landmarks)
    driving_lips = extract_lip_landmarks(driving_landmarks)
    output_lips = extract_lip_region(output)
    return F.mse_loss(output_lips, driving_lips - source_lips + output_lips)

def extract_lip_landmarks(landmarks):
    return landmarks[48:]

def extract_lip_region(image):
    return image[:, :, 2*image.shape[2]//3:, image.shape[3]//4:3*image.shape[3]//4]

def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, perceptual_loss, stage):
    generator.train()
    discriminator.train()
    num_epochs = STAGE_1_EPOCHS if stage == 1 else STAGE_2_EPOCHS
    
    for epoch in range(num_epochs):
        running_loss_G, running_loss_D = 0.0, 0.0
        progress_bar = tqdm(dataloader, desc=f"Stage {stage} - Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (source_images, driving_images, source_landmarks, driving_landmarks) in enumerate(progress_bar):
            source_images, driving_images = source_images.to(device), driving_images.to(device)
            source_landmarks, driving_landmarks = source_landmarks.to(device), driving_landmarks.to(device)
            batch_size = source_images.size(0)
            
            optimizer_G.zero_grad()
            generated_images = generator(source_images, driving_images)
            
            loss_adv = calculate_adversarial_loss(generated_images, discriminator)
            loss_perceptual = calculate_perceptual_loss(generated_images, driving_images, perceptual_loss)
            loss_stitching = calculate_stitching_loss(generated_images, source_images)
            loss_eye = calculate_eye_retargeting_loss(generated_images, source_landmarks, driving_landmarks)
            loss_lip = calculate_lip_retargeting_loss(generated_images, source_landmarks, driving_landmarks)
            
            loss_G = loss_adv + loss_perceptual if stage == 1 else loss_adv + loss_perceptual + loss_stitching + loss_eye + loss_lip
            
            loss_G.backward()
            optimizer_G.step()
            
            optimizer_D.zero_grad()
            
            real_pred = discriminator(driving_images)
            fake_pred = discriminator(generated_images.detach())
            
            loss_D.backward()
            optimizer_D.step()
            
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            
            progress_bar.set_postfix({'G_Loss': running_loss_G / (batch_idx + 1), 'D_Loss': running_loss_D / (batch_idx + 1)})
        
        print(f"Stage {stage} - Epoch [{epoch+1}/{num_epochs}], G_Loss: {running_loss_G/len(dataloader):.4f}, D_Loss: {running_loss_D/len(dataloader):.4f}")
        
        # Log to wandb
        wandb.log({
            f"Stage_{stage}_G_Loss": running_loss_G/len(dataloader),
            f"Stage_{stage}_D_Loss": running_loss_D/len(dataloader),
            "epoch": epoch
        })
    
    return running_loss_G / len(dataloader), running_loss_D / len(dataloader)

def validate(generator, dataloader, device, perceptual_loss):
    generator.eval()
    val_loss = 0.0
    with torch.no_grad():
        for source_images, driving_images, _, _ in tqdm(dataloader, desc="Validating"):
            source_images, driving_images = source_images.to(device), driving_images.to(device)
            generated_images = generator(source_images, driving_images)
            loss = calculate_perceptual_loss(generated_images, driving_images, perceptual_loss)
            val_loss += loss.item()
    return val_loss / len(dataloader)

def main():
    wandb.init(project="live-portrait-generation", entity="your-entity-name")
    
    print(f"Using device: {device}")
    
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    dataset = CelebADataset(transform=transform)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    generator = LivePortrait().to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss = VGGPerceptualLoss().to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)
    
    wandb.watch(generator)
    wandb.watch(discriminator)
    
    best_val_loss = float('inf')
    
    print("✨ Starting Stage 1 training...")
    for epoch in range(STAGE_1_EPOCHS):
        train_loss_G, train_loss_D = train(generator, discriminator, train_dataloader, optimizer_G, optimizer_D, device, perceptual_loss, stage=1)
        val_loss = validate(generator, val_dataloader, device, perceptual_loss)
        
        scheduler_G.step()
        scheduler_D.step()
        
        print(f"Epoch [{epoch+1}/{STAGE_1_EPOCHS}], Train G_Loss: {train_loss_G:.4f}, Train D_Loss: {train_loss_D:.4f}, Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "Stage_1_Train_G_Loss": train_loss_G,
            "Stage_1_Train_D_Loss": train_loss_D,
            "Stage_1_Val_Loss": val_loss,
            "epoch": epoch
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), "best_generator_stage1.pth")
            torch.save(discriminator.state_dict(), "best_discriminator_stage1.pth")
    
    print("🎉 Stage 1 training completed.")
    
    print("✨ Starting Stage 2 training...")
    for epoch in range(STAGE_2_EPOCHS):
        train_loss_G, train_loss_D = train(generator, discriminator, train_dataloader, optimizer_G, optimizer_D, device, perceptual_loss, stage=2)
        val_loss = validate(generator, val_dataloader, device, perceptual_loss)
        
        scheduler_G.step()
        scheduler_D.step()
        
        print(f"Epoch [{epoch+1}/{STAGE_2_EPOCHS}], Train G_Loss: {train_loss_G:.4f}, Train D_Loss: {train_loss_D:.4f}, Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "Stage_2_Train_G_Loss": train_loss_G,
            "Stage_2_Train_D_Loss": train_loss_D,
            "Stage_2_Val_Loss": val_loss,
            "epoch": epoch + STAGE_1_EPOCHS
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), "best_generator_stage2.pth")
            torch.save(discriminator.state_dict(), "best_discriminator_stage2.pth")
    
    print("🎉 Stage 2 training completed.")
    
    # Final save
    torch.save(generator.state_dict(), "final_generator.pth")
    torch.save(discriminator.state_dict(), "final_discriminator.pth")
    print("💾 Final models saved successfully.")
    
    wandb.finish()

if __name__ == "__main__":
    main()