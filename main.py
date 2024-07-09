import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
STAGE_1_EPOCHS = 80
STAGE_2_EPOCHS = 20

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

class CelebADataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.dataset = load_dataset("huggan/celeba-faces", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_image = self.dataset[idx]["image"]
        driving_image = self.dataset[np.random.randint(len(self.dataset))]["image"]
        
        if self.transform:
            source_image = self.transform(source_image)
            driving_image = self.transform(driving_image)
        
        return source_image, driving_image

class AppearanceFeatureExtractor(nn.Module):
    def __init__(self):
        super(AppearanceFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
    
    def forward(self, x):
        return self.features(x)

class MotionExtractor(nn.Module):
    def __init__(self):
        super(MotionExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(256 * 32 * 32, 256)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class WarpingModule(nn.Module):
    def __init__(self):
        super(WarpingModule, self).__init__()
        self.conv1 = nn.Conv2d(2048 + 256, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 2, kernel_size=3, padding=1)
    
    def forward(self, appearance_features, motion_features):
        b, c, h, w = appearance_features.shape
        motion_features = motion_features.view(b, 256, 1, 1).expand(-1, -1, h, w)
        combined = torch.cat([appearance_features, motion_features], dim=1)
        x = F.relu(self.conv1(combined))
        flow = self.conv2(x)
        return flow

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.tanh(self.conv5(x))
        return x

class StitchingModule(nn.Module):
    def __init__(self):
        super(StitchingModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(126, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 65)
        )
    
    def forward(self, x_s, x_d):
        input_features = torch.cat([x_s.view(-1), x_d.view(-1)])
        return self.mlp(input_features)

class EyeRetargetingModule(nn.Module):
    def __init__(self):
        super(EyeRetargetingModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(66, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 63)
        )
    
    def forward(self, x_s, c_s_eyes, c_d_eyes):
        input_features = torch.cat([x_s.view(-1), c_s_eyes, c_d_eyes.unsqueeze(0)])
        return self.mlp(input_features)

class LipRetargetingModule(nn.Module):
    def __init__(self):
        super(LipRetargetingModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(65, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 63)
        )
    
    def forward(self, x_s, c_s_lip, c_d_lip):
        input_features = torch.cat([x_s.view(-1), c_s_lip.unsqueeze(0), c_d_lip.unsqueeze(0)])
        return self.mlp(input_features)

class LivePortrait(nn.Module):
    def __init__(self):
        super(LivePortrait, self).__init__()
        self.appearance_extractor = AppearanceFeatureExtractor()
        self.motion_extractor = MotionExtractor()
        self.warping_module = WarpingModule()
        self.decoder = Decoder()
        self.stitching_module = StitchingModule()
        self.eye_retargeting_module = EyeRetargetingModule()
        self.lip_retargeting_module = LipRetargetingModule()
    
    def forward(self, source_image, driving_image, c_s_eyes, c_d_eyes, c_s_lip, c_d_lip):
        appearance_features = self.appearance_extractor(source_image)
        motion_features = self.motion_extractor(driving_image)
        
        flow = self.warping_module(appearance_features, motion_features)
        warped_features = self.warp(appearance_features, flow)
        
        stitching_offset = self.stitching_module(appearance_features, warped_features)
        eye_offset = self.eye_retargeting_module(appearance_features, c_s_eyes, c_d_eyes)
        lip_offset = self.lip_retargeting_module(appearance_features, c_s_lip, c_d_lip)
        
        final_features = warped_features + stitching_offset + eye_offset + lip_offset
        output_image = self.decoder(final_features)
        
        return output_image
    
    def warp(self, features, flow):
        b, c, h, w = features.shape
        grid = self.get_grid(b, h, w).to(features.device)
        final_grid = (grid + flow).permute(0, 2, 3, 1)
        return F.grid_sample(features, final_grid, mode='bilinear', padding_mode='border')
    
    @staticmethod
    def get_grid(b, h, w):
        xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
        yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
        xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
        yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        return grid

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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
        super(VGGPerceptualLoss, self).__init__()
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
            x = block(x)
            y = block(y)
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
    mask[:, :, :shape[2]//3, :] = 0  # Assume top 1/3 is face
    return mask

def calculate_eye_retargeting_loss(output, c_s_eyes, c_d_eyes):
    eye_region = extract_eye_region(output)
    target_openness = c_d_eyes - c_s_eyes
    current_openness = calculate_eye_openness(eye_region)
    return F.mse_loss(current_openness, target_openness)

def extract_eye_region(image):
    return image[:, :, :image.shape[2]//3, image.shape[3]//4:3*image.shape[3]//4]

def calculate_eye_openness(eye_region):
    return eye_region.mean()

def calculate_lip_retargeting_loss(output, c_s_lip, c_d_lip):
    lip_region = extract_lip_region(output)
    target_openness = c_d_lip - c_s_lip
    current_openness = calculate_lip_openness(lip_region)
    return F.mse_loss(current_openness, target_openness)

def extract_lip_region(image):
    return image[:, :, 2*image.shape[2]//3:, image.shape[3]//4:3*image.shape[3]//4]

def calculate_lip_openness(lip_region):
    return lip_region.mean()

def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, perceptual_loss, stage):
    generator.train()
    discriminator.train()
    
    num_epochs = STAGE_1_EPOCHS if stage == 1 else STAGE_2_EPOCHS
    
    for epoch in range(num_epochs):
        running_loss_G = 0.0
        running_loss_D = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Stage {stage} - Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (source_images, driving_images) in enumerate(progress_bar):
            source_images, driving_images = source_images.to(device), driving_images.to(device)
            batch_size = source_images.size(0)
            
            # Generate random conditions
            c_s_eyes = torch.rand(2).to(device)
            c_d_eyes = torch.rand(1).to(device)
            c_s_lip = torch.rand(1).to(device)
            c_d_lip = torch.rand(1).to(device)
            
            # Train Generator
            optimizer_G.zero_grad()
            generated_images = generator(source_images, driving_images, c_s_eyes, c_d_eyes, c_s_lip, c_d_lip)
            
            loss_adv = calculate_adversarial_loss(generated_images, discriminator)
            loss_perceptual = calculate_perceptual_loss(generated_images, driving_images, perceptual_loss)
            loss_stitching = calculate_stitching_loss(generated_images, source_images)
            loss_eye = calculate_eye_retargeting_loss(generated_images, c_s_eyes, c_d_eyes)
            loss_lip = calculate_lip_retargeting_loss(generated_images, c_s_lip, c_d_lip)
            
            if stage == 1:
                loss_G = loss_adv + loss_perceptual
            else:
                loss_G = loss_adv + loss_perceptual + loss_stitching + loss_eye + loss_lip
            
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            real_pred = discriminator(driving_images)
            fake_pred = discriminator(generated_images.detach())
            
            loss_D_real = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
            loss_D_fake = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
            loss_D = (loss_D_real + loss_D_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            
            progress_bar.set_postfix({
                'G_Loss': running_loss_G / (batch_idx + 1),
                'D_Loss': running_loss_D / (batch_idx + 1)
            })
        
        print(f"Stage {stage} - Epoch [{epoch+1}/{num_epochs}], G_Loss: {running_loss_G/len(dataloader):.4f}, D_Loss: {running_loss_D/len(dataloader):.4f}")
    
    return running_loss_G / len(dataloader), running_loss_D / len(dataloader)

def main():
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = CelebADataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    generator = LivePortrait().to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss = VGGPerceptualLoss().to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Stage 1 training
    print("Starting Stage 1 training...")
    stage1_loss_G, stage1_loss_D = train(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, perceptual_loss, stage=1)
    print(f"Stage 1 training completed. Final G_Loss: {stage1_loss_G:.4f}, D_Loss: {stage1_loss_D:.4f}")
    
    # Stage 2 training
    print("Starting Stage 2 training...")
    stage2_loss_G, stage2_loss_D = train(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, perceptual_loss, stage=2)
    print(f"Stage 2 training completed. Final G_Loss: {stage2_loss_G:.4f}, D_Loss: {stage2_loss_D:.4f}")
    
    # Save the trained models
    torch.save(generator.state_dict(), "liveportrait_generator.pth")
    torch.save(discriminator.state_dict(), "liveportrait_discriminator.pth")
    print("Models saved successfully.")

if __name__ == "__main__":
    main()