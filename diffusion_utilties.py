import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

# 残差卷积块
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # 检查 input channel 和 output channel 是否一样
        self.same_channels = in_channels == out_channels

        # 是否使用残差连接
        self.is_res = is_res

        # 第一层卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

        # 第二层卷积层  
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 使用残差连接
        if self.is_res:
            
            x1 = self.conv1(x)

            x2 = self.conv2(x1)

            # 如果 input channel 和 output channel 一样，直接相加
            if self.same_channels:
                out = x + x2
            else:
                # 若维度不匹配，则在添加残差连接前应用 1x1 卷积层进行维度对齐。 将x的通道数调整为与x2相同的通道数。
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2

            # Normalize output tensor
            return out / 1.414

        # 不使用残差连接
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels

        

# 上采样模块
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        
        layers = [
            # 转置卷积层（Transposed Convolution）​
            # 输入：batch_size=1, channel=64, size=4x4 (1x64x4x4)
            # 输出尺寸：1x32x8x8
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        
        x = torch.cat((x, skip), 1)
        
        x = self.model(x)
        return x

# 下采样模块    
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        layers = [
            ResidualConvBlock(in_channels, out_channels), 
            ResidualConvBlock(out_channels, out_channels), 
            nn.MaxPool2d(2)]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()

        # 通过全连接层（线性层）将 ​标量输入（如时间步、类别标签）​ 转换为高维嵌入向量
    
        self.input_dim = input_dim  # 输入特征维度（如时间步为1，类别标签为n_cfeat）
        
        # 定义网络层结构
        layers = [
            nn.Linear(input_dim, emb_dim),  # 第一线性层: 升维/降维
            nn.GELU(),                      # 高斯误差线性单元激活函数（比ReLU更平滑）
            nn.Linear(emb_dim, emb_dim),    # 第二线性层: 保持维度不变的非线性变换
        ]
        
        # 构建PyTorch序列模型
        self.model = nn.Sequential(*layers)  # *操作符解包列表为独立参数

    def forward(self, x):
        # 输入形状适配 (支持任意batch_size)
        # 例如: 当输入为标量时间步时，将形状从 [B] 转换为 [B, 1]
        x = x.view(-1, self.input_dim)  # -1 表示自动推导batch维度
        
        # 执行嵌入变换（示例: 输入[B,1] → 输出[B,emb_dim]）
        return self.model(x)
    
def unorm(x):
    """
    对单张图像进行逐通道归一化，将每个通道的像素值缩放到 [0, 1] 范围。
    输入形状: (h, w, 3) ，即高度、宽度、RGB通道。
    """
    # 计算每个通道的最大值和最小值（在高度和宽度维度上）
    xmax = x.max((0, 1))  # 形状: (3,)，每个通道的最大值
    xmin = x.min((0, 1))  # 形状: (3,)，每个通道的最小值
    # 逐通道归一化：(x - min) / (max - min)
    return (x - xmin) / (xmax - xmin)

def norm_all(store, n_t, n_s):
    """
    对存储的多时间步、多样本数据进行批量归一化。
    输入:
        - store: 形状为 (n_t, n_s, h, w, 3) 的四维数组，
          表示 `n_t` 个时间步和 `n_s` 个样本的图像数据。
        - n_t: 时间步总数。
        - n_s: 样本总数。
    输出: 归一化后的数据，形状与输入相同。
    """
    nstore = np.zeros_like(store)
    # 遍历每个时间步和样本，逐个归一化
    for t in range(n_t):
        for s in range(n_s):
            nstore[t, s] = unorm(store[t, s])  # 调用 unorm 处理单个图像
    return nstore

def norm_torch(x_all):
    """
    对PyTorch格式的图像张量进行批量归一化。
    输入形状: (n_samples, 3, h, w) 即样本数、RGB通道、高度、宽度。
    输出形状: 归一化后的数据，形状与输入相同。
    """
    # 转换为NumPy数组并处理
    x = x_all.cpu().numpy()  # 形状: (n_samples, 3, h, w)
    # 计算每个样本每个通道的最大值和最小值（在高度和宽度维度上）
    xmax = x.max((2, 3))      # 形状: (n_samples, 3)
    xmin = x.min((2, 3))      # 形状: (n_samples, 3)
    # 扩展维度以支持广播
    xmax = np.expand_dims(xmax, (2, 3))  # 形状: (n_samples, 3, 1, 1)
    xmin = np.expand_dims(xmin, (2, 3))  # 形状: (n_samples, 3, 1, 1)
    # 逐样本、逐通道归一化
    nstore = (x - xmin) / (xmax - xmin)
    return nstore

def gen_tst_context(n_cfeat):
    """
    Generate test context vectors
    """
    vec = torch.tensor([
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0]]      # human, non-human, food, spell, side-facing
    )
    return len(vec), vec

def plot_grid(x,n_sample,n_rows,save_dir,w):
    # x:(n_sample, 3, h, w)
    ncols = n_sample//n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)  # curiously, nrow is number of columns.. or number of items in the row.
    save_image(grid, save_dir + f"run_image_w{w}.png")
    print('saved image at ' + save_dir + f"run_image_w{w}.png")
    return grid

def plot_sample(x_gen_store,n_sample,nrows,save_dir, fn,  w, save=False):
    ncols = n_sample//nrows
    sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow
    
    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
        return plots
    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0]) 
    plt.close()
    if save:
        ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print('saved gif at ' + save_dir + f"{fn}_w{w}.gif")
    return ani


class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape
                
    # Return the number of images in the dataset
    def __len__(self):
        return len(self.sprites)
    
    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape

transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])
