import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from dataset import my_Dataset, Vocab_builder
from models.v1_CNN_BiLSTM import v1_Recognizer
from transforms import transform

word2Index, Index2word = Vocab_builder("data/DataSet/UHWR/train.txt")

val_Dataset = my_Dataset(
    label_path="data/DataSet/UHWR/overfit_test.txt",
    word2Index=word2Index,
    Index2word=Index2word,
    dataset_path="data/DataSet/UHWR/",
    transform=transform
)

image, label, real_width = val_Dataset[0]
image = image.unsqueeze(0)

model = v1_Recognizer()
model.load_state_dict(torch.load("checkpoints/overfit_test/epoch_411.pt", map_location='cpu'))
model.eval()

with torch.no_grad():
    log_probs = model(image)

prediction = torch.argmax(log_probs, dim=2)  # shape: (batch=1, timesteps)
pred_indices = prediction[0].tolist()

# CTC collapse: merge consecutive repeats, then strip blanks
blank_idx = 117
collapsed = []
prev = None
for idx in pred_indices:
    if idx != prev:
        collapsed.append(idx)
    prev = idx
collapsed = [idx for idx in collapsed if idx != blank_idx]

predicted_str = ''.join(Index2word[idx] for idx in collapsed)
true_str = ''.join(Index2word[idx] for idx in label)

print(f"Raw prediction indices: {pred_indices}")
print(f"Predicted (decoded, collapsed): {predicted_str}")
print(f"True label (decoded): {true_str}")