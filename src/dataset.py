from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class my_Dataset(Dataset):
    def __init__(self,label_path,word2Index,Index2word,dataset_path,transform):
        self.word2Index=word2Index
        self.Index2word=Index2word
        self.data=[]
        self.transform=transform

        with open(label_path,encoding='utf-8') as file:
            skipped_incoPath=0
            skipped_widTooBig=0
            for line in file:
                cleaned_line=line.strip()
                words=cleaned_line.split('\t')
                full_path=dataset_path+words[0]
                if os.path.exists(full_path):
                    if Image.open(full_path).size[0]<=1400:  
                      self.data.append((full_path,words[1]))
                    else:
                       skipped_widTooBig+=1
                else:
                     skipped_incoPath+=1
            print(f'Skipped files due to incorrect path: { skipped_incoPath}')
            print(f'Skipped files due to incorrect path: { skipped_widTooBig}')
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        (image_path,label)=self.data[idx]

        img=Image.open(image_path)
        int_labels=[]
        for char in label:
            int_labels.append(self.word2Index[char])

        img=self.transform(img)
        return img,int_labels
    

def Vocab_builder(path):

    training_data_set=set()
 
    #set of unique chars in dataset
    with open(path,encoding='utf-8') as file:
      for line in file:
        l=line.strip()
        c=l.split('\t')
        for w in c[1]:
            training_data_set.add(w)

    #going with the set from training data to generate the word2Index and Index2word dictionary
    word2Index={}
    Index2word={}
    sorted_list=sorted(training_data_set) #ensures same order each time 
    for (index,char) in enumerate(sorted_list):
        word2Index[char]=index
        Index2word[index]=char
    
    orig_len=len(word2Index)
    word2Index['<Pad>']=orig_len
    Index2word[orig_len]='<Pad>'
    return word2Index,Index2word


def collate_fn(batch,word2Index):
    images, label_lists = zip(*batch)

    image_batch = torch.stack(images)

    label_lengths = [len(l) for l in label_lists]
    max_len = max(label_lengths)

    padded_labels = torch.full((len(label_lists), max_len), len(word2Index)-1, dtype=torch.long)
    for i, label in enumerate(label_lists):
        padded_labels[i, :len(label)] = torch.tensor(label, dtype=torch.long)

    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return image_batch, padded_labels, label_lengths

