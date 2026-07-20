import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import Vocab_builder
from PIL import Image

word2Index, Index2word = Vocab_builder("data/DataSet/UHWR/train.txt")

with open("data/DataSet/UHWR/train.txt", encoding='utf-8') as file:
    first_line = file.readline().strip()

original_text = "انکار نہیں کیا جاسکتا کہ آج کے دور میں ملکی ترقی"
print("original text:", original_text)
original_text=original_text[::-1]
encoded = [word2Index[c] for c in original_text]
print("encoded:", encoded)

decoded = ''.join([Index2word[i] for i in encoded])
print("decoded back:", decoded)

print("match:", original_text == decoded)

img=Image.open("data/DataSet/UHWR/Dataset/images/10136.jpg")
img.show()