from torch.utils.data import Dataset
from PIL import Image

class my_Dataset(Dataset):
    def __init__(self,label_path,word2Index,Index2word,dataset_path):
        self.word2Index=word2Index
        self.Index2word=Index2word
        self.data=[]

        with open(label_path,encoding='utf-8') as file:
            for line in file:
                cleaned_line=line.strip()
                words=cleaned_line.split('\t')
                self.data.append((dataset_path+words[0],words[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        (image_path,label)=self.data[idx]

        img=Image.open(image_path)
        int_labels=[]
        for char in label:
            int_labels.append(self.word2Index[char])

        return img,int_labels
    

def Vocab_builder():

    training_data_set=set()
 
    #set of unique chars in dataset
    with open("data/DataSet/UHWR/train.txt",encoding='utf-8') as file:
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
    
    return word2Index,Index2word
