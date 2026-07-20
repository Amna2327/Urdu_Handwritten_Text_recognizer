from PIL import Image
widths=[]

with open("data/DataSet/UHWR/train.txt",encoding='utf-8') as file:
    for line in file:
        cleaned=line.strip()
        paths=cleaned.split('\t')
        image_path="data/DataSet/UHWR/"+paths[0]

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