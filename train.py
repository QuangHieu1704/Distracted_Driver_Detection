from dataset import *
from models.VGG16 import VGG16
from models.VGG19 import VGG19
from models.ResNet import ResNet50, ResNet101, ResNet152
import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import pandas as pd
from torchsummary import summary

  
if __name__ == "__main__":
    train_path = "Distracted_Driver_Detection\\Dataset\\train"
    val_path = "Distracted_Driver_Detection\\Dataset\\val"
    test_path = "Distracted_Driver_Detection\\Dataset\\test"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 50
    batch_size = 64
    learning_rate = 0.0001

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(train_path, transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=2)

    valid_dataset = torchvision.datasets.ImageFolder(val_path, transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, num_workers=2)

    test_dataset = torchvision.datasets.ImageFolder(test_path, transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers=2)

    # Define model
    # model = VGG16()
    # model = VGG19()
    model = ResNet50(num_classes = 10)

    print(summary(model, (3, 224, 224), device="cpu"))

    if torch.cuda.is_available():
        model.cuda()

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

    # # Training and validating
    # for epoch in range(epochs):
    #     model.train()
    #     train_loss = 0.0
    #     for i, (images, labels) in enumerate(train_loader):
    #         images = images.to(device)
    #         labels = labels.to(device)

    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         # Forward pass
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()

    #         if i % 100 == 0:
    #             print(f"Epoch [{epoch+1}/{epochs}], iter [{i+1}/{len(train_loader)}], loss: {loss.item()}")

    #     train_loss = train_loss / len(train_loader)

    #     # Validating
    #     y_predicted = []
    #     y_ground_truth = []
    #     with torch.no_grad():
    #         model.eval()
    #         for images, labels in valid_loader:
    #             images = images.to(device)
    #             y_ground_truth.extend(labels.tolist())
    #             labels = labels.to(device)
    #             outputs = model(images)

    #             _, predicted_cls = torch.max(outputs, 1) 
    #             y_predicted.extend(predicted_cls.tolist())
    #     val_accuracy = accuracy_score(y_ground_truth, y_predicted)
    #     val_precision = precision_score(y_ground_truth, y_predicted, average='macro')
    #     val_recall = recall_score(y_ground_truth, y_predicted, average='macro')
    #     val_f1score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    #     print (f'Epoch [{epoch+1}/{epochs}], Train_Loss: {train_loss}, Val_Acc: {val_accuracy}, Val_Precision: {val_precision}, Val_Recall: {val_recall}, Val_F1score: {val_f1score}')
    #     torch.save(model.state_dict(), os.path.join("Distracted_Driver_Detection/weights/VGG16", f"VGG16_epoch{epoch+1}.pth"))

    #     model.train()
    #     train_loss = 0.0
