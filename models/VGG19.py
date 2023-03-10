import torch

class VGG19(torch.nn.Module):
  def __init__(self):
    super(VGG19, self).__init__()
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU())
    self.layer2 = torch.nn.Sequential(
      torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2))

    self.layer3 = torch.nn.Sequential(
      torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(128),
      torch.nn.ReLU())
    self.layer4 = torch.nn.Sequential(
      torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(128),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2))
      
    self.layer5 = torch.nn.Sequential(
      torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(256),
      torch.nn.ReLU())
    self.layer6 = torch.nn.Sequential(
      torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(256),
      torch.nn.ReLU())
    self.layer7 = torch.nn.Sequential(
      torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(256),
      torch.nn.ReLU())
    self.layer8 = torch.nn.Sequential(
      torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(256),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
    

    self.layer9 = torch.nn.Sequential(
      torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512),
      torch.nn.ReLU())
    self.layer10 = torch.nn.Sequential(
      torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512),
      torch.nn.ReLU())
    self.layer11 = torch.nn.Sequential(
      torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512),
      torch.nn.ReLU())
    self.layer12 = torch.nn.Sequential(
      torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size = 2, stride = 2))

    self.layer13 = torch.nn.Sequential(
      torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512),
      torch.nn.ReLU())
    self.layer14 = torch.nn.Sequential(
      torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512),
      torch.nn.ReLU())
    self.layer15 = torch.nn.Sequential(
      torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512),
      torch.nn.ReLU())  
    self.layer16 = torch.nn.Sequential(
      torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size = 2, stride = 2))

    
    self.fc = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(7 * 7 * 512, 512),
      torch.nn.ReLU(),
      torch.nn.Linear(512, 10))
    

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.layer7(out)
    out = self.layer8(out)
    out = self.layer9(out)
    out = self.layer10(out)
    out = self.layer11(out)
    out = self.layer12(out)
    out = self.layer13(out)
    out = self.layer14(out)
    out = self.layer15(out)
    out = self.layer16(out)
    out = self.fc(out)

    return out