import torch
from torch.utils.data import DataLoader
from dataset import my_Dataset,Vocab_builder,collate_fn
from models.v1_CNN_BiLSTM import v1_Recognizer
from transforms import transform

def train_one_epoch(model,train_loader,optimizer,criterion):
    model.train()
    train_loss=0
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

      train_loss+=loss.item()
      if c%100==0:
         print(f"Batch number: {c}, loss(per batch): {loss.item()}")

    return train_loss

def validate(model,val_loader,criterion):
  model.eval()
  total_loss=0
  for image_batch, labels, label_lengths in val_loader:
      with torch.no_grad():
        log_probs=model(image_batch)
        log_probs=log_probs.transpose(0,1)
        input_length=torch.full((image_batch.size(0),),200,dtype=torch.long)
        loss=criterion(log_probs,labels,input_length,label_lengths)
        total_loss+=loss.item()
  return total_loss

def main():
  word2Index,Index2word=Vocab_builder("data/DataSet/UHWR/train.txt")

  train_Dataset=my_Dataset(
   label_path="data/DataSet/UHWR/train.txt",
   word2Index=word2Index,
   Index2word=Index2word,
   dataset_path="data/DataSet/UHWR/",
   transform=transform
   )

  val_Dataset=my_Dataset(
   label_path="data/DataSet/UHWR/val.txt",
   word2Index=word2Index,
   Index2word=Index2word,
   dataset_path="data/DataSet/UHWR/",
   transform=transform
   )

  train_loader=DataLoader(train_Dataset,batch_size=8,shuffle=True,collate_fn=lambda b: collate_fn(b,word2Index))
  val_loader=DataLoader(val_Dataset,batch_size=8,shuffle=True,collate_fn=lambda b: collate_fn(b,word2Index))

  model=v1_Recognizer()
  criterion=torch.nn.CTCLoss(blank=117)
  optimizer=torch.optim.Adam(model.parameters(), lr=0.0003)

  epochs=1
  for i in range(epochs):
     train_loss=train_one_epoch(model,train_loader,optimizer,criterion)
     val_loss=validate(model,val_loader,criterion)
     print(f'epoch: {i+1}, train_loss={train_loss/len(train_loader):.4f}')
     print(f'epoch: {i+1}, val_loss={val_loss/len(val_loader):.4f}')
     torch.save(model.state_dict(), f"checkpoints/epoch_{i+1}.pt")

if __name__=="__main__":
   main()
