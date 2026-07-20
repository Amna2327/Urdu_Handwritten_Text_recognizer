import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # adjust if needed

from dataset import my_Dataset, Vocab_builder
from transforms import transform

def num_adjacent_repeats(label_str):

    """Count adjacent repeated characters — CTC needs a blank between each pair."""
    count = 0

    for i in range(1, len(label_str)):
        if label_str[i] == label_str[i - 1]:
            count += 1
    return count

def main():
    word2Index, Index2word = Vocab_builder("data/DataSet/UHWR/train.txt")
    dataset = my_Dataset(
        label_path="data/DataSet/UHWR/train.txt",
        word2Index=word2Index,
        Index2word=Index2word,
        dataset_path="data/DataSet/UHWR/",
        transform=transform,
    )

    bad_samples = []
    for idx in range(len(dataset)):
        img_path, raw_label = dataset.data[idx]
        # raw_label is the original (non-reversed) string from your label file
        _, int_labels, real_width = dataset[idx]
        label_len = len(int_labels)
        repeats = num_adjacent_repeats(raw_label)
        min_required = label_len + repeats

        if real_width < min_required:
            bad_samples.append({
                "path": img_path,
                "label": raw_label,
                "label_len": label_len,
                "repeats": repeats,
                "min_required_timesteps": min_required,
                "actual_input_length": real_width,
            })

    total = len(dataset)
    print(f"Total samples: {total}")
    print(f"Samples that WOULD produce inf CTC loss: {len(bad_samples)} "
          f"({100 * len(bad_samples) / total:.2f}%)")

    if bad_samples:
        print("\nFirst 10 problem samples:")
        for s in bad_samples[:10]:
            print(
                f"  label='{s['label']}' label_len={s['label_len']} "
                f"repeats={s['repeats']} needs>={s['min_required_timesteps']} "
                f"but input_length={s['actual_input_length']}  path={s['path']}"
            )

if __name__ == "__main__":

    main()