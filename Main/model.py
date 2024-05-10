import torch

class Model(torch.nn.Module):
    def __init__(self, backbone, neck, head):
        super(Model, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
    
    def forward(self, x):
        features, feature_maps = self.backbone(x)
        features = features.permute(0, 2, 3, 1)
        features = self.neck(features, feature_maps)
        features = features.permute(0, 3, 1, 2)
        detection_output = self.head(features)
        return detection_output