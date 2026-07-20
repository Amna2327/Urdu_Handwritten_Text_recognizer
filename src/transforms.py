from PIL import Image
import torchvision.transforms.functional as TF
import torch

def transform(image):
    image=image.convert('L')
    width,height=image.size
    new_width=int(width*(64/height))
    image=image.resize((new_width,64))

    img_tensor=TF.to_tensor(image)
    return torch.nn.functional.pad(img_tensor,(0,1400-img_tensor.shape[-1]),value=1.0),new_width//8


