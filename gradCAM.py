import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import pickle
from object_detection import get_classes

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        class_names = get_classes()
        model_name = "beerchallenge_resnet50_7brands.pth"

        # define the resnet152
        self.resnet = resnet50(pretrained=True)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, len(class_names))
        self.resnet.load_state_dict(torch.load(model_name))

        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1,
                                      self.resnet.layer2,
                                      self.resnet.layer3,
                                      self.resnet.layer4)

        # average pooling layer
        self.avgpool = self.resnet.avgpool

        # classifier
        self.classifier = self.resnet.fc

        # gradient placeholder
        self.gradient = None

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def forward(self, x):
        # extract the features
        x = self.features(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)

        return x


def heatmap(img_location, heatmap_location, class_int=None, opacity=0.8):
    class_names = get_classes()
    # init the resnet
    resnet = ResNet()
    # set the evaluation mode
    _ = resnet.eval()

    #open image
    img = Image.open(img_location)
    #img = TF.to_tensor(img)
    #img.unsqueeze_(0)

    test_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224), #doordat image wordt verkleint tot 224x224 image (op random plek), is uitput steeds anders
        #transforms.CenterCrop(224),  # pak altijd het center van de image
        # transforms.RandomHorizontalFlip(), #ook dit kan (kleine) impact hebben op de output
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])  # normalize images for R, G, B (both mean and SD)

    img = test_transforms(img)
    # add 1 dimension to tensor
    img=img.unsqueeze(0)
    # forward pass
    pred = resnet(img)

    #pred.argmax(dim=1)  # prints tensor([2])
    sm = torch.nn.Softmax(dim=1)  # use softmax to convert tensor values to probs (dim = do columns (0) or rows (1) have to sum up to 1?)
    probabilities = sm(pred)

    # get the gradient of the output with respect to the parameters of the model
    if class_int==None:
        pred[:, pred.argmax()].backward()
    else:
        pred[:, class_int].backward()

    # pull the gradients out of the model
    gradients = resnet.get_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = resnet.get_activations(img).detach()
    # len(activations[0])

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    #plt.matshow(heatmap.squeeze())

    # make the heatmap to be a numpy array
    heatmap = heatmap.numpy()

    # interpolate the heatmap
    img = cv2.imread(img_location)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #
    superimposed_img = (heatmap * opacity) + img
    #save
    cv2.imwrite(heatmap_location, superimposed_img)
    return Image.open(heatmap_location), probabilities, class_names[pred.argmax()]
