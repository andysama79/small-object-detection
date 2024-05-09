import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.nn.modules.pooling import AdaptiveAvgPool2d
import timm
# from swin_transformer import SwinTransformer
from functools import partial

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# print("Data loaded")

# load pretrained Swin

#! switch to mmdetection
model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True)
print("Model created")
print(model)

# class ClassifierHead(nn.Module):
    # def __init__(self, in_features, out_features):
    #     super(ClassifierHead, self).__init__()
    #     self.global_pool = AdaptiveAvgPool2d(output_size=(1, 1))
    #     self.drop = nn.Dropout(p=0.0)
    #     self.fc = nn.Linear(in_features, out_features)
    #     self.flatten = nn.Identity()
        
    # def forward(self, x):
    #     x = x.permute(0, 3, 1, 2)
    #     x = self.global_pool(x)
    #     x = self.drop(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)
    #     return x
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
# print(type(model))

# state_dict = torch.load('../pretrained_models/swin_small_patch4_window7_224.pth')
# model.load_state_dict(state_dict)
# modify output
num_classes = 10
in_features = model.head.in_features
model.head = nn.Linear(in_features, in_features)
# model.head = ClassifierHead(in_features, num_classes)
# print("New Head: ", model.head)

# # loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # train
# num_epochs = 10
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("...using device: ", device)
# model.to(device)

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
    
#     train_loss = running_loss / len(train_loader.dataset)
#     train_acc = 100. * correct / total

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')

# # evaluation
# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

# test_acc = 100. * correct / total
# print(f'Test Accuracy: {test_acc:.4f}')

import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.resize((224, 224))
    img = torch.from_numpy(img).unsqueeze(0)
    return img

img = preprocess_image("../assets/annotated_image.jpg")

print(img.shape)

pred = (model(img))

print(pred.shape)