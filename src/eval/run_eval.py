# evaluate.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # adjust if needed

import torch
from models.v1_CNN_BiLSTM import v1_Recognizer
from eval import evaluate_cer
from dataset import my_Dataset, Vocab_builder, collate_fn
from transforms import transform
from torch.utils.data import DataLoader

def main():
    word2Index, Index2word = Vocab_builder("data/DataSet/UHWR/train.txt")

    val_Dataset = my_Dataset(
        label_path="data/DataSet/UHWR/val.txt",
        word2Index=word2Index,
        Index2word=Index2word,
        dataset_path="data/DataSet/UHWR/",
        transform=transform
    )
    val_loader = DataLoader(val_Dataset, batch_size=8, shuffle=False, collate_fn=lambda b: collate_fn(b, word2Index))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = v1_Recognizer()
    model.load_state_dict(torch.load("checkpoints/best_model_3layer_BiLSTM.pt", map_location=device))
    model.to(device)

    overall_cer, preds, refs = evaluate_cer(model, val_loader, Index2word, device, method="beam", beam_width=10)
    print(f"CER: {overall_cer:.4f}")
    for p, r in zip(preds[:15], refs[:15]):
        print(f"pred: {p}\nref:  {r}\n")

if __name__ == "__main__":
    main()