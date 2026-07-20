from PIL import Image
widths=[]
labels=[]

with open("data/DataSet/UHWR/train.txt",encoding='utf-8') as file:
    for line in file:
        cleaned=line.strip()
        paths=cleaned.split('\t')
        image_path="data/DataSet/UHWR/"+paths[0]
        labels.append(paths[1])
        repeats = 0

        try:
            img=Image.open(image_path)
            widths.append(img.size[0])
        except FileNotFoundError:
            continue

widths.sort()
n=len(widths)
print(n)

print(f"Max width: {widths[-1]}")
print(f"Min width: {widths[0]}")
print(f"average width: {(sum(widths)/n)}")
print(f"median: {(widths[n//2])}")

for cutoff in [1000, 1100, 1200, 1300, 1400, 1500, 1600]:
    count_above = sum(1 for w in widths if w > cutoff)
    percent_above = (count_above / n) * 100
    print(f"cutoff {cutoff}: {count_above} images above ({percent_above:.2f}%)")


max_required = 0
worst_label = ""
for label in labels:
    repeats = 0
    for i in range(1, len(label)):
        if label[i] == label[i-1]:
            repeats += 1

    required = len(label) + repeats

    if required > max_required:
        max_required = required
        worst_label = label

print("worst-case required CTC length:", max_required)
print("that label's raw length:", len(worst_label))
print("number of consecutive repeats in it:", max_required - len(worst_label))

labels=[len(l) for l in labels]
labels.sort()
no_of_labels=len(labels)
print(f"Max lable size: {labels[-1]}")
print(f"Min lable size: {labels[0]}")
print(f"average width: {(sum(labels)/no_of_labels)}")
print(f"median: {(labels[no_of_labels//2])}")