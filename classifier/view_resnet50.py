import torchvision.models as models

# Load the pre-trained ResNet-50 model
resnet_model = models.resnet50()

# Print the model architecture
print(resnet_model)
print()
print(resnet_model.fc.in_features)