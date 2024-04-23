import urllib.request
import timm
import torch.nn as nn
import torch

class MobileNetV3TL(nn.Module):
    def __init__(self, path=None):
        super(MobileNetV3TL, self).__init__()
        if path is not None:
            self.load_state_dict(torch.load(path))
        else:
            self.model = timm.create_model('mobilenetv3_large_100', pretrained=True)
            self.model.eval()
        
    def forward(self, x):
        return self.model(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    import os
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    
    model = MobileNetV3TL()

    # load image
    dogfile = "dog.jpg"
    if not os.path.exists(dogfile):
        import urllib
        urllib.request.urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", dogfile)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    img = Image.open(dogfile).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    # predict
    with torch.no_grad():
        out = model(tensor)
    probas = torch.nn.functional.softmax(out[0], dim=0)
    print(probas.shape)

    # print top 5
    classfile = "imagenet_classes.txt"
    if not os.path.exists(classfile):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", classfile)
    with open(classfile) as f:
        classes = [line.strip() for line in f.readlines()]
    top5_prob, top5_catid = torch.topk(probas, 5)
    for i in range(5):
        print(f"Class: {classes[top5_catid[i]]}, Probability: {top5_prob[i].item()}")
