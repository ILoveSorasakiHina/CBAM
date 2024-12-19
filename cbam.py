import torch
import torch.nn as nn

class channel_block(nn.Module):
    def __init__(self,channel = 3,reduction = 16):
        super().__init__()
        #AdaptiveAvgPool2d可以指定輸出大小，當參數為1，等於全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            #inplace=True代表允許直接修改輸入數據，節省內存開銷
            #假如為False，ReLU就要先用另一個變數儲存運算結果，然後再賦值給輸入數值
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        out = avg_out + max_out
        out = self.sigmoid(out)
        return x * out.expand_as(x) 
    
class spatial_block(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #沿著通道做平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #沿著通道做最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        print(max_out)
        #疊合池化後的特徵圖
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)

        return x * out
    
channel = channel_block()
print(channel)
input_tensor = torch.randn(1, 3, 32, 32)
output_tensor = channel(input_tensor)
print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)

spatial = spatial_block()
print(spatial)
output_tensor = spatial(output_tensor)
print("Output shape:", output_tensor.shape)

