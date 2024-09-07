'''
    这段代码是我用来测试GetRasterBand(1)的
    GetRasterBand(i) 方法是 GDAL 库中用于访问栅格数据集中特定波段数据的方法。
    这里的 i 是波段的索引号，通常从1开始计数（而不是从0开始）。每个波段都包含了图像在特定波长范围内的反射或辐射信息。
'''

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

# 替换为你的 TIFF 文件路径
tif_path = 'data/2_95_sat.tif'

# 使用 GDAL 打开 TIFF 文件
dataset = gdal.Open(tif_path)

# 检查文件是否成功打开
if dataset is None:
    print("文件打开失败，请检查路径是否正确")
else:
    # 获取第一个波段
    band = dataset.GetRasterBand(1)

    # 读取波段数据为 NumPy 数组
    data = band.ReadAsArray()

    # 显示图像
    plt.imshow(data, cmap='gray')  # 使用灰度色图显示图像
    plt.colorbar()  # 显示颜色条
    plt.show()