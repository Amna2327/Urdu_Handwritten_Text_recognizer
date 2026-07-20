#since we keep gettign CTC collapse and a plateaued loss curve, might as well confirm if our model is even correct in the first place or not
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from dataset import my_Dataset,Vocab_builder,collate_fn
from models.v1_CNN_BiLSTM import v1_Recognizer
from transforms import transform

def train_one_epoch(model,train_loader,optimizer,criterion,device):
    model.train()
    train_loss=0
    for image_batch, labels, label_lengths,input_length in train_loader:
      optimizer.zero_grad()

      image_batch=image_batch.to(device)
      labels=labels.to(device)
      input_length=input_length.to(device)

      log_probs=model(image_batch)
      log_probs=log_probs.transpose(0,1)

      loss=criterion(log_probs,labels,input_length,label_lengths)
      loss.backward()
      print(model.CNN_Encoder.ConvLayer1.weight.grad.norm())
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
      optimizer.step()

      train_loss+=loss.item()

    return train_loss

def main():
  word2Index,Index2word=Vocab_builder("data/DataSet/UHWR/train.txt")

  train_Dataset=my_Dataset(
   label_path="data/DataSet/UHWR/overfit_test.txt",
   word2Index=word2Index,
   Index2word=Index2word,
   dataset_path="data/DataSet/UHWR/",
   transform=transform
   )

  train_loader=DataLoader(train_Dataset,batch_size=5,shuffle=True,collate_fn=lambda b: collate_fn(b,word2Index))
  model=v1_Recognizer()
  criterion=torch.nn.CTCLoss(blank=117, zero_infinity=True)
  optimizer=torch.optim.Adam(model.parameters(), lr=0.0003)

  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model=model.to(device)

  epochs=500
  for i in range(epochs):
    train_loss=train_one_epoch(model,train_loader,optimizer,criterion,device)
    avg_loss=train_loss/len(train_loader)
    print(f'epoch: {i+1}, train_loss={avg_loss:.4f}')

    if i%50==0:
      torch.save(model.state_dict(), f"checkpoints/overfit_test/epoch_{i+1}.pt")

    if avg_loss<0.01:
       print(f"Loss below threshold, stopping early at epoch {i+1}")
       torch.save(model.state_dict(), f"checkpoints/overfit_test/epoch_{i+1}.pt")
       break
       
     

if __name__=="__main__":
   main()
