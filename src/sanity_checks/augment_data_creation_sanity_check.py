import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # adjust if needed

from PIL import Image
from dataset import my_Dataset, Vocab_builder
from transforms import train_transform
import torchvision.transforms.functional as TF

word2Index, Index2word = Vocab_builder("data/DataSet/UHWR/train.txt")
ds = my_Dataset(
    label_path="data/DataSet/UHWR/train.txt",
    word2Index=word2Index,
    Index2word=Index2word,
    dataset_path="data/DataSet/UHWR/",
    transform=train_transform
)

# Save 5 augmented versions of the same sample to eyeball
for i in range(5):
    img_tensor, label, real_width = ds[1]
    img = TF.to_pil_image(img_tensor)
    img.save(f"aug_sample_{i}.png")