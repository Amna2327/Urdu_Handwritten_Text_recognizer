import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from dataset import my_Dataset,Vocab_builder
from models.v1_CNN_BiLSTM import v1_Recognizer
from transforms import transform


word2Index,Index2word=Vocab_builder("data/DataSet/UHWR/train.txt")
train_Dataset=my_Dataset(
   label_path="data/DataSet/UHWR/overfit_test.txt",
   word2Index=word2Index,
   Index2word=Index2word,
   dataset_path="data/DataSet/UHWR/",
   transform=transform
   )
model=v1_Recognizer()
model.load_state_dict(torch.load("checkpoints/overfit_test/epoch_100.pt", map_location='cpu'))
model.eval()
with torch.no_grad():
    img_a, _ = train_Dataset[0]
    img_b, _ = train_Dataset[3]
    
    out_a = model.CNN_Encoder(img_a.unsqueeze(0))
    out_b = model.CNN_Encoder(img_b.unsqueeze(0))

    print(f"Mean value of output_a: {out_a.mean()}")
    print(f"Min value of output_a: {out_a.min()}")
    print(f"Max value of output_a: {out_a.max()}")

    print(f"Mean value of output_b: {out_b.mean()}")
    print(f"Min value of output_b: {out_b.min()}")
    print(f"Max value of output_b: {out_b.max()}")

    out_a=out_a.squeeze(2)
    out_b=out_b.squeeze(2)
    out_a=out_a.transpose(1,2)
    out_b=out_b.transpose(1,2)

    per_timestep_output_a,NONE=model.BiLSTM(out_a)
    per_timestep_output_b,NONE=model.BiLSTM(out_b)

    diff_CNN = (out_a - out_b).abs().mean()
    diff_BiLSTM_bw_img = (per_timestep_output_a - per_timestep_output_b).abs().mean()
    diff_BiLSTM_wi_img = (per_timestep_output_a[:,109,:] - per_timestep_output_a[:,174,:]).abs().mean()
    print("mean absolute difference between CNN outputs:", diff_CNN.item())
    print("mean absolute difference between BiLSTM outputs:", diff_BiLSTM_bw_img.item())
    print("mean absolute difference between timestamps of BiLSTM of the same image:", diff_BiLSTM_wi_img.item())