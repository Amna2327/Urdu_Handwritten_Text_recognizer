# src/eval.py
from jiwer import cer
from decode import ctc_greedy_decode, ctc_beam_search_decode
import torch

def evaluate_cer(model, val_loader, Index2word, device, blank=117, method="beam", beam_width=10):
    model.eval()
    all_preds, all_refs = [], []

    with torch.no_grad():
        for image_batch, labels, label_lengths, input_length in val_loader:
            image_batch = image_batch.to(device)
            log_probs = model(image_batch)

            if method == "beam":
                preds = ctc_beam_search_decode(log_probs, Index2word, blank=blank, beam_width=beam_width)
            else:
                preds = ctc_greedy_decode(log_probs, Index2word, blank=blank)
            all_preds.extend(preds)

            for i, length in enumerate(label_lengths):
                label_seq = labels[i, :length].tolist()
                all_refs.append(''.join(Index2word[idx] for idx in label_seq))

    return cer(all_refs, all_preds), all_preds, all_refs