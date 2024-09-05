import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

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

def predict_image(model, image_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('I')  # 16-bit grayscale
    image = image.resize((256, 256), Image.BILINEAR)  # Resize to match model input
    image = np.array(image, dtype=np.float32)
    image = (image / 65535.0) * 255.0  # Normalize
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        output = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).cpu().numpy()[0]  # Remove batch dimension

    # Convert predicted_class to PIL Image and save
    pred_img = Image.fromarray(predicted_class.astype(np.uint8), mode='L')
    pred_img = pred_img.resize((1516, 1514), Image.NEAREST)  # Resize to original size
    pred_img.save(output_path)

if __name__ == "__main__":
    # Load pre-trained model weights
    model = UNet()
    model.load_state_dict(torch.load('unet_model.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    # Predict and save results for a single image
    test_image_path = './test_images/test.png'
    output_image_path = './test_images/prediction.png'
    predict_image(model, test_image_path, output_image_path)
    print(f"Prediction saved to '{output_image_path}'")
