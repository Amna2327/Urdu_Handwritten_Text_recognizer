# confusion_analysis.py
# Run this after eval to see exactly which characters your model confuses,
# drops, or hallucinates. Reuses your existing eval pipeline to get pred/ref
# pairs, then does a proper edit-distance alignment (not just raw string diff)
# to attribute each error to a specific character-level operation.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))

import torch
from collections import defaultdict, Counter
from torch.utils.data import DataLoader

from dataset import my_Dataset, Vocab_builder, collate_fn
from models.v1_CNN_BiLSTM import v1_Recognizer
from transforms import transform
from eval import evaluate_cer


def align_and_diff(ref, pred):
    """
    Computes the optimal (minimum edit distance) alignment between ref and pred
    strings using Wagner-Fischer DP, then backtraces to classify each operation:
    match, substitution, insertion (extra char in pred), deletion (missing char).

    Returns a list of ('match'|'sub'|'ins'|'del', ref_char_or_None, pred_char_or_None)
    """
    n, m = len(ref), len(pred)
    # dp[i][j] = edit distance between ref[:i] and pred[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion (char in ref, missing from pred)
                    dp[i][j - 1],      # insertion (extra char in pred)
                    dp[i - 1][j - 1],  # substitution
                )

    # Backtrace from (n, m) to (0, 0)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == pred[j - 1]:
            ops.append(('match', ref[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(('sub', ref[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('del', ref[i - 1], None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(('ins', None, pred[j - 1]))
            j -= 1
        else:
            # Shouldn't happen, but guard against infinite loop
            break

    ops.reverse()
    return ops


def build_confusion_report(refs, preds, top_n=30):
    substitutions = Counter()   # (ref_char, pred_char) -> count
    deletions = Counter()       # ref_char -> count (model missed this char)
    insertions = Counter()      # pred_char -> count (model hallucinated this char)
    char_totals = Counter()     # ref_char -> total occurrences in ground truth
    char_errors = Counter()     # ref_char -> total error occurrences (sub + del)

    for ref, pred in zip(refs, preds):
        ops = align_and_diff(ref, pred)
        for op, r, p in ops:
            if op == 'match':
                char_totals[r] += 1
            elif op == 'sub':
                substitutions[(r, p)] += 1
                char_totals[r] += 1
                char_errors[r] += 1
            elif op == 'del':
                deletions[r] += 1
                char_totals[r] += 1
                char_errors[r] += 1
            elif op == 'ins':
                insertions[p] += 1
                # insertions don't correspond to a real ref char, tracked separately

    print("=" * 70)
    print(f"TOP {top_n} CHARACTER SUBSTITUTIONS (ref -> pred : count)")
    print("=" * 70)
    for (r, p), count in substitutions.most_common(top_n):
        print(f"  '{r}' -> '{p}'  : {count}")

    print()
    print("=" * 70)
    print(f"TOP {top_n} DELETIONS (model drops this ref char : count)")
    print("=" * 70)
    for r, count in deletions.most_common(top_n):
        print(f"  '{r}'  : {count}")

    print()
    print("=" * 70)
    print(f"TOP {top_n} INSERTIONS (model hallucinates this char : count)")
    print("=" * 70)
    for p, count in insertions.most_common(top_n):
        print(f"  '{p}'  : {count}")

    print()
    print("=" * 70)
    print(f"TOP {top_n} CHARACTERS BY ERROR RATE (min 5 occurrences)")
    print("  -> these are the characters most worth targeting with")
    print("     augmentation or synthetic data generation")
    print("=" * 70)
    error_rates = []
    for char, total in char_totals.items():
        if total >= 5:
            err = char_errors.get(char, 0)
            error_rates.append((char, err / total, err, total))
    error_rates.sort(key=lambda x: x[1], reverse=True)
    for char, rate, err, total in error_rates[:top_n]:
        print(f"  '{char}'  error_rate={rate:.2%}  ({err}/{total} occurrences)")

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "char_totals": char_totals,
        "char_errors": char_errors,
    }


def save_confusion_csv(report, path="confusion_matrix.csv"):
    import csv
    with open(path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["type", "ref_char", "pred_char", "count"])
        for (r, p), count in report["substitutions"].most_common():
            writer.writerow(["substitution", r, p, count])
        for r, count in report["deletions"].most_common():
            writer.writerow(["deletion", r, "", count])
        for p, count in report["insertions"].most_common():
            writer.writerow(["insertion", "", p, count])
    print(f"\nFull confusion data saved to {path}")


def main():
    word2Index, Index2word = Vocab_builder("data/DataSet/UHWR/train.txt")

    val_Dataset = my_Dataset(
        label_path="data/DataSet/UHWR/val.txt",
        word2Index=word2Index,
        Index2word=Index2word,
        dataset_path="data/DataSet/UHWR/",
        transform=transform
    )
    val_loader = DataLoader(val_Dataset, batch_size=8, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, word2Index))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = v1_Recognizer()

    # CHANGE THIS to whichever checkpoint you want to analyze
    checkpoint_path = "checkpoints/best_model_3layer_BiLSTM.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    print(f"Analyzing checkpoint: {checkpoint_path}\n")

    overall_cer, preds, refs = evaluate_cer(model, val_loader, Index2word, device,
                                             method="beam", beam_width=10)
    print(f"Overall CER: {overall_cer:.4f}\n")

    report = build_confusion_report(refs, preds, top_n=30)
    save_confusion_csv(report, path="confusion_matrix.csv")


if __name__ == "__main__":
    main()