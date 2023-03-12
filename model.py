import torch 
import torch.nn as nn

class FC(nn.Module):
    def __init__(self, channel_size, flatten_size=15960, cls_num=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,30,kernel_size=(1,25)),
            nn.Conv2d(30,30, kernel_size=(channel_size, 1)),
            nn.AvgPool2d(kernel_size=(1, 75), stride=15),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, cls_num)
        )
        
        
    def forward(self, x):
        
        feat = self.conv(x)
        pred = self.fc(feat)
        
        return feat, pred


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16,1),
        )
    
    def forward(self, x):
        return self.model(x)

#loss_dis
def Loss_dis(pred_t, pred_s):
    return torch.mean(torch.square(pred_t-1)+torch.square(pred_s)) / 2 


if __name__ == "__main__":
    print(__name__)