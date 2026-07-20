#bias terms check
"""import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from models.v1_CNN_BiLSTM import v1_Recognizer

model = v1_Recognizer()
model.load_state_dict(torch.load("checkpoints/overfit_test/epoch_100.pt", map_location='cpu'))
model.eval()

print(model.output_layer.bias)
print("blank index (117) bias:", model.output_layer.bias[117].item())
print("max bias overall:", model.output_layer.bias.max().item())
print("which index has max bias:", model.output_layer.bias.argmax().item())"""

#top predictions across a few timestamps to check whether it even considers other than class 117
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from models.v1_CNN_BiLSTM import v1_Recognizer
from dataset import Vocab_builder, my_Dataset
from transforms import transform

word2Index, Index2word = Vocab_builder("data/DataSet/UHWR/train.txt")

model = v1_Recognizer()
model.load_state_dict(torch.load("checkpoints/overfit_test/epoch_100.pt", map_location='cpu'))
model.eval()

train_Dataset = my_Dataset(
    label_path="data/DataSet/UHWR/overfit_test.txt",
    word2Index=word2Index,
    Index2word=Index2word,
    dataset_path="data/DataSet/UHWR/",
    transform=transform
)

image, _ = train_Dataset[0]
image = image.unsqueeze(0)

with torch.no_grad():
    log_probs = model(image)

# check a few timesteps
for t in [10, 50, 100, 150]:
    scores = log_probs[0, t, :]
    top5_values, top5_indices = torch.topk(scores, 5)
    print(f"timestep {t}:")
    print("  top 5 classes:", top5_indices.tolist())
    print("  top 5 log-prob scores:", top5_values.tolist())