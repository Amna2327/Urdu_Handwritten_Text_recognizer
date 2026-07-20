import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

# 2 time-steps, batch size 1, 4 classes (0,1,2 = real chars, 3 = blank)
criterion = torch.nn.CTCLoss(blank=3)

# "confident and correct" prediction: strongly predicts 0, then 1
log_probs_correct = torch.log(torch.tensor([
    [[0.9, 0.03, 0.03, 0.04]],  # time-step 1: mostly class 0
    [[0.03, 0.9, 0.03, 0.04]],  # time-step 2: mostly class 1
]))

# "always blank" prediction
log_probs_blank = torch.log(torch.tensor([
    [[0.03, 0.03, 0.03, 0.91]],
    [[0.03, 0.03, 0.03, 0.91]],
]))

labels = torch.tensor([0, 1])
input_lengths = torch.tensor([2])
label_lengths = torch.tensor([2])

loss_correct = criterion(log_probs_correct, labels, input_lengths, label_lengths)
loss_blank = criterion(log_probs_blank, labels, input_lengths, label_lengths)

print("loss when prediction is correct:", loss_correct.item())
print("loss when prediction is always blank:", loss_blank.item())