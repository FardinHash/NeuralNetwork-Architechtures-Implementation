# import libraries
from efficientnet_pytorch import EfficientNetV2
from torch import nn
import torch.nn.functional as F

# define input shape
input_shape = (224, 224, 3)

# create the base model
base_model = EfficientNetV2.from_pretrained("efficientnet-b0", num_classes = 1000)

#create the new model
class newModel(nn.Module):
    def __init__(self):
        super(newModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.classifier.in_features, num_classes)
    def forward(self, x):
        x = self.base_model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x

#model
model = newModel()
