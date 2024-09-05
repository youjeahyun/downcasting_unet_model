import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Custom Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, size=(256, 256)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.size = size
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(img_path).convert('I')  # 16-bit grayscale
        label = Image.open(label_path).convert('L')  # 8-bit grayscale

        # Resize image and label
        image = image.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.uint8)

        # Normalize the image to [0, 255]
        image = (image / 65535.0) * 255.0  # Convert 16-bit to 8-bit
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Convert label to a format suitable for segmentation
        label = np.where(np.isin(label, [0, 1, 2, 3]), label, 0)  # Ensure only 0, 1, 2, 3 are kept

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label, dtype=torch.long)

        return image, label

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = SegmentationDataset('./images', './labels', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# U-Net Model Definition
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.encoder1 = self.contracting_block(1, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)

        # Bottleneck
        self.bottleneck = self.contracting_block(256, 512)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder1 = self.expansive_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.expansive_block(256, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self.expansive_block(128, 64)

        self.final_conv = nn.Conv2d(64, 4, kernel_size=1)  # 4 classes: 0, 1, 2, 3

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def expansive_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        bottleneck = self.bottleneck(enc3)

        up1 = self.upconv1(bottleneck)
        if up1.size(2) != enc3.size(2) or up1.size(3) != enc3.size(3):
            enc3 = F.interpolate(enc3, size=up1.size()[2:], mode='bilinear', align_corners=True)
        dec1 = self.decoder1(torch.cat([up1, enc3], dim=1))

        up2 = self.upconv2(dec1)
        if up2.size(2) != enc2.size(2) or up2.size(3) != enc2.size(3):
            enc2 = F.interpolate(enc2, size=up2.size()[2:], mode='bilinear', align_corners=True)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))

        up3 = self.upconv3(dec2)
        if up3.size(2) != enc1.size(2) or up3.size(3) != enc1.size(3):
            enc1 = F.interpolate(enc1, size=up3.size()[2:], mode='bilinear', align_corners=True)
        dec3 = self.decoder3(torch.cat([up3, enc1], dim=1))

        return self.final_conv(dec3)

# 학습 루프
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 클래스 가중치를 적용한 손실 함수
class_weights = torch.tensor([1.0, 2.0, 5.0, 1.0], dtype=torch.float32).to(device)  # 예시 가중치
def weighted_cross_entropy(outputs, labels):
    return F.cross_entropy(outputs, labels, weight=class_weights)

def train_one_epoch():
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)

        # 출력과 레이블의 크기를 동일하게 맞춤
        if outputs.size(2) != labels.size(1) or outputs.size(3) != labels.size(2):
            labels = F.interpolate(labels.unsqueeze(1).float(), size=outputs.size()[2:], mode='nearest').squeeze(1).long()

        # 손실 계산
        loss = weighted_cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# 조기 종료를 포함한 학습
def train_with_early_stopping(max_epochs=100, threshold=0.03):
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}")
        avg_loss = train_one_epoch()
        print(f"Training Loss: {avg_loss}")

        if avg_loss < threshold:
            print(f"로스가 {avg_loss:.4f}로 임계값 {threshold:.4f} 이하이므로 조기 종료합니다.")
            break

train_with_early_stopping()

# 학습된 모델 저장
torch.save(model.state_dict(), 'unet_model.pth')
print("모델이 'unet_model.pth'로 저장되었습니다.")
# Visualization function
def visualize_samples(dataset, num_samples=3):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    for i in range(num_samples):
        img, label = dataset[i]
        img = img.squeeze(0).numpy()  # Remove batch dimension
        label = label.squeeze(0).numpy()

        # Normalize 16-bit image to 8-bit
        img_normalized = (img / np.max(img)) * 255.0
        img_8bit = np.clip(img_normalized, 0, 255).astype(np.uint8)

        # Display image
        axes[i, 0].imshow(img_8bit, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        # Display label
        axes[i, 1].imshow(label, cmap='jet', vmin=0, vmax=3)
        axes[i, 1].set_title('Label')
        axes[i, 1].axis('off')

        # Display prediction
        output = model(torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
        output = F.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
        axes[i, 2].imshow(prediction, cmap='jet', vmin=0, vmax=3)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.show()
# 학습 후 샘플 시각화
visualize_samples(dataset, num_samples=3)

def predict_and_save(model, dataset, image_dir, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, (image, _) in enumerate(dataset):
            # 원본 이미지 파일 이름 가져오기
            image_filename = dataset.image_files[i]
            image_id = os.path.splitext(image_filename)[0]  # 확장자 제거

            # 예측을 위한 이미지 처리
            image = image.unsqueeze(0).to(device, dtype=torch.float32)  # 배치 차원 추가
            output = model(image)

            # 소프트맥스를 적용하여 확률을 얻고, argmax로 예측 클래스 얻기
            output = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).cpu().numpy()[0]  # 배치 차원 제거

            # 예측 클래스를 PIL 이미지로 변환하고 원본 크기로 리사이즈
            pred_img = Image.fromarray(predicted_class.astype(np.uint8), mode='L')
            pred_img = pred_img.resize((1516, 1514), Image.NEAREST)  # 원본 크기로 리사이즈

            # 원본 이미지와 같은 이름으로 예측 저장
            pred_img.save(os.path.join(output_dir, f'{image_id}.png'))

# 예측 결과를 원본 이미지 파일 이름으로 저장
predict_and_save(model, dataset, './images', './predictions')
print("예측 결과가 './predictions'에 저장되었습니다.")