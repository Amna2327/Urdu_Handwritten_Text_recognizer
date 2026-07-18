import torch
from dataset import my_Dataset,Vocab_builder
from models.v1_CNN_BiLSTM import v1_Recognizer
from transforms import transform

word2Index,Index2word=Vocab_builder("data/DataSet/UHWR/train.txt")

val_Dataset=my_Dataset(
   label_path="data/DataSet/UHWR/val.txt",
   word2Index=word2Index,
   Index2word=Index2word,
   dataset_path="data/DataSet/UHWR/",
   transform=transform
)

image,label=val_Dataset[75]
image=image.unsqueeze(0)

model=v1_Recognizer()
model.load_state_dict(torch.load("checkpoints/epoch_5.pt", map_location='cpu'))
model.eval()
log_probs=model(image)

prediction=torch.argmax(log_probs,dim=2)
print(f"predicted: {prediction}")
print(f"True output: {label}")