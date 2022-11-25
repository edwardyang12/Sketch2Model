import torch
import torch.nn as nn
from nets.utils import MiniBatchDiscrimination, GaussianNoise

# final layer is one hot prediction
class OrigDiscriminator(nn.Module):
    def __init__(self, channels=3, feat_map=64, batch=20):
        super(OrigDiscriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 4, feat_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 8, feat_map * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 16, feat_map * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 32, feat_map * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 64, feat_map * 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 128, feat_map * 256, 4, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (feat_map*8) x 4 x 4
            nn.Conv2d(feat_map * 256, 3, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

# final layer is patch prediction
class PatchGAN(nn.Module):
    def __init__(self, channels=3, feat_map=64, batch=20):
        super(PatchGAN, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map*2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 4, feat_map*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 8, feat_map*16, 4, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 16, 3, 4, 1, 1, bias=False),
        )

    def forward(self, input):
        return self.main(input)


# final layer is patch prediction
class PatchMiniBatch(nn.Module):
    def __init__(self, channels=3, feat_map=64, batch=20):
        super(PatchMiniBatch, self).__init__()

        # 15,15 determined from the patch size
        self.mbd1 = MiniBatchDiscrimination(feat_map*8, feat_map*8, feat_map*4, batch) # insert before last layer
        self.main = nn.Sequential(
            # input is actually (channels) x 64 x 64 patches
            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map*2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map*4, feat_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (feat_map*8) x 4 x 4
            nn.Conv2d(feat_map * 8, feat_map*8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.feat_map = feat_map
        self.batch = batch
        self.out = nn.Conv2d(feat_map * 16, 3, 4, 1, 1, bias=False)

    def forward(self, input):
        x = self.main(input)
        B,C,H,W = x.shape
        mbd_layer = self.mbd1(torch.mean(x,dim=[2,3])).view(B,C,1,1)
        mbd_layer = mbd_layer.expand(-1,-1,H,W)
        return self.out(torch.cat((x,mbd_layer),dim=1))

class PatchMiniBatchNoise(nn.Module):
    def __init__(self, channels=3, feat_map=64, batch=20):
        super(PatchMiniBatchNoise, self).__init__()

        self.mbd1 = MiniBatchDiscrimination(feat_map*8, feat_map*8, feat_map*4, batch) # insert before last layer
        self.main = nn.Sequential(

            GaussianNoise(),
            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(),
            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(),
            nn.Conv2d(feat_map*2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(),
            nn.Conv2d(feat_map * 4, feat_map*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(),
            nn.Conv2d(feat_map * 8, feat_map*8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.feat_map = feat_map
        self.batch = batch
        self.out = nn.Conv2d(feat_map * 16, 3, 4, 1, 1, bias=False)

    def forward(self, input):
        x = self.main(input)
        B,C,H,W = x.shape
        mbd_layer = self.mbd1(torch.mean(x,dim=[2,3])).view(B,C,1,1)
        mbd_layer = mbd_layer.expand(-1,-1,H,W)
        return self.out(torch.cat((x,mbd_layer),dim=1))

class PatchMiniBatchNoiseLabel(nn.Module):
    def __init__(self, channels=3, feat_map=64, batch=20, classes = 10):
        super(PatchMiniBatchNoiseLabel, self).__init__()

        self.mbd1 = MiniBatchDiscrimination(feat_map*8, feat_map*8, feat_map*4, batch) # insert before last layer
        self.main = nn.Sequential(

            GaussianNoise(),
            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(),
            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(),
            nn.Conv2d(feat_map*2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(),
            nn.Conv2d(feat_map * 4, feat_map*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(),
            nn.Conv2d(feat_map * 8, feat_map*8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),


        )
        self.feat_map = feat_map
        self.batch = batch
        self.out = nn.Conv2d(feat_map * 16, 3, 4, 1, 1, bias=False)
        self.classes = classes

        self.classifier = nn.Sequential(
            # need to flatten before using
            # 246016 is number of features?
            nn.Linear(115200, classes),
        )

    def forward(self, input):
        x = self.main(input)
        label = torch.flatten(x,start_dim=1, end_dim=-1)
        pred = self.classifier(label)
        B,C,H,W = x.shape
        mbd_layer = self.mbd1(torch.mean(x,dim=[2,3])).view(B,C,1,1)
        mbd_layer = mbd_layer.expand(-1,-1,H,W)
        return self.out(torch.cat((x,mbd_layer),dim=1)), pred