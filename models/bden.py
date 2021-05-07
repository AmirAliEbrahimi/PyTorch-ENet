import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import  BinarizeConv2d,InputScale,SignumActivation,BinarizeTransposedConv2d

class BDEN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.ratioInfl=16
        self.numOfClasses=num_classes

        self.FrontLayer = nn.Sequential(
            InputScale(),
            BinarizeConv2d(3, int(4*self.ratioInfl), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            SignumActivation(),
            BinarizeConv2d(int(4*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            SignumActivation(),

            BinarizeConv2d(int(4*self.ratioInfl), int(8*self.ratioInfl), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(8*self.ratioInfl)),
            SignumActivation(),
            BinarizeConv2d(int(8*self.ratioInfl), int(8*self.ratioInfl), kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(int(8*self.ratioInfl)),
            SignumActivation(),
            BinarizeConv2d(int(8*self.ratioInfl), int(16*self.ratioInfl), kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(int(16*self.ratioInfl)),
            SignumActivation(),

            BinarizeTransposedConv2d(int(16*self.ratioInfl), int(16*self.ratioInfl), kernel_size=3, stride=2 , padding=1,output_padding=1),
            nn.BatchNorm2d(int(16*self.ratioInfl)),
            SignumActivation(),
            BinarizeConv2d(int(16*self.ratioInfl), int(8*self.ratioInfl), kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(int(8*self.ratioInfl)),
            SignumActivation(),

            BinarizeTransposedConv2d(int(8*self.ratioInfl), int(8*self.ratioInfl), kernel_size=3, padding=1,stride=2,output_padding=1),
            nn.BatchNorm2d(int(8*self.ratioInfl)),
            SignumActivation(),
            BinarizeConv2d(int(8*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            SignumActivation()
        )

        self.TailLayer = nn.Sequential(
            BinarizeConv2d(int(4*self.ratioInfl), self.numOfClasses, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(self.numOfClasses),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.FrontLayer(x)
        x = self.TailLayer(x)
        return x
