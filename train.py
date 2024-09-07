import torch.nn as nn
import torch
import gdal
import numpy as np
from torch.utils.data import Dataset, DataLoader

class UNet(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(UNet, self).__init__()
        # enc 表示编码器（下采样路径）， dec 表示解码器（上采样路径）。
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.center = self.conv_block(512, 1024)
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64,out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    # 每个卷积块由两个卷积层组成，每个卷积层后面跟着 ReLU 激活函数和批量归一化。
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        center = self.center(self.pool(enc4))

        dec4 = self.dec4(torch.cat([enc4, self.up(center)], 1))
        dec3 = self.dec3(torch.cat([enc3, self.up(dec4)], 1))
        dec2 = self.dec2(torch.cat([enc2, self.up(dec3)], 1))
        dec1 = self.dec1(torch.cat([enc1, self.up(dec2)], 1))
        final = self.final(dec1).squeeze()

        return torch.sigmoid(final)


class RSDataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, images_dir, labels_dir):
        self.images = self.read_multiband_images(images_dir)
        self.labels = self.read_singleband_labels(labels_dir)

    # 读取多波段图像数据
    def read_multiband_images(self, images_dir):
        images = []                                         # 始化一个空列表 images，用于存储所有读取的图像数据。
        for image_path in images_dir:                          # 遍历 images_dir 列表中的每个元素。每个元素都是一个图像文件的路径。
            rsdl_data = gdal.Open(image_path)              #  使用 GDAL 库的 Open 函数打开当前循环中的图像文件
            images.append(np.stack([rsdl_data .GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0))
        return images


    # 读取多波段图像标签
    def read_singleband_labels(self, labels_dir):
        labels = []
        for label_path in labels_dir:
            rsdl_data = gdal.Open(label_path)
            labels.append(rsdl_data .GetRasterBand(1).ReadAsArray())
        return labels

    # 返回数据集大小：
    def __len__(self):
        return len(self.images)

    # 得到数据内容和标签：
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)



#  指定图像和标签的位置
images_dir = ['data/2_95_sat.tif', 'data/2_96_sat.tif',  'data/2_97_sat.tif', 'data/2_98_sat.tif', 'data/2_976_sat.tif']
labels_dir =['data/2_95_mask.tif', 'data/2_96_mask.tif',  'data/2_97_mask.tif', 'data/2_98_mask.tif', 'data/2_976_mask.tif']

# 创建一个 RSDataset 实例，这个实例负责加载图像和标签数据
dataset = RSDataset(images_dir, labels_dir)
# 使用 DataLoader 创建一个数据加载器，它将从 dataset 中批量加载数据
trainloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 创建一个 UNet 模型实例，输入通道数为3（假设是RGB图像），输出通道数为1（假设是二分类分割任务，例如掩膜分割）
model = UNet(3, 1)
# 定义损失函数为二元交叉熵损失（Binary Cross-Entropy Loss），适用于二分类问题。
criterion = nn.BCELoss()
# 定义优化器为 Adam，这是一种基于梯度下降的优化算法，用于更新模型的权重。学习率设置为0.001。
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 置训练的总轮数（epochs）为50
num_epochs=50

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        images = images.float()
        labels = labels.float()/255.0
        outputs = model(images)
        labels = labels.squeeze(0)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

torch.save(model.state_dict(), 'models_building_50.pth')