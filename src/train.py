import torch
from torch.utils.data import DataLoader
from dataset import my_Dataset,Vocab_builder,collate_fn
from models.v1_CNN_BiLSTM import v1_Recognizer
from transforms import transform

word2Index,Index2word=Vocab_builder("data/DataSet/UHWR/train.txt")

train_Dataset=my_Dataset(
   label_path="data/DataSet/UHWR/train.txt",
   word2Index=word2Index,
   Index2word=Index2word,
   dataset_path="data/DataSet/UHWR/",
   transform=transform
   )

train_loader=DataLoader(train_Dataset,batch_size=8,shuffle=True,collate_fn=lambda b: collate_fn(b,word2Index))

model=v1_Recognizer()
criterion=torch.nn.CTCLoss(blank=117)
optimizer=torch.optim.Adam(model.parameters(), lr=0.0003)

epoch=5

for i in range(epoch):
    model.train()
    total_loss=0
    c=0

    for image_batch, labels, label_lengths in train_loader:
      optimizer.zero_grad()
      c+=1

      log_probs=model(image_batch)
      log_probs=log_probs.transpose(0,1)

      input_length=torch.full((image_batch.size(0),),200,dtype=torch.long)

      loss=criterion(log_probs,labels,input_length,label_lengths)
      loss.backward()
      optimizer.step()

      total_loss+=loss.item()

    print(f'epoch: {i+1}, loss={total_loss/len(train_loader):.4f}')
