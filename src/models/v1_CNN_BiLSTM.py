import torch.nn as nn
import torch

class CNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvLayer1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.BatchNorm1=nn.BatchNorm2d(32)

        self.ConvLayer2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.BatchNorm2=nn.BatchNorm2d(64)

        self.ConvLayer3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.BatchNorm3=nn.BatchNorm2d(128)

        self.ConvLayer4=nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.BatchNorm4=nn.BatchNorm2d(128)

        self.ConvLayer5=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.BatchNorm5=nn.BatchNorm2d(256)

        self.ConvLayer6=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.BatchNorm6=nn.BatchNorm2d(256)

        self.LeakyRelU=nn.LeakyReLU()
        self.MaxPool2d_dimhalf=nn.MaxPool2d((2,2))
        self.MaxPool2d_heighthalf=nn.MaxPool2d((2,1))

        self.Dropout_01=nn.Dropout2d(0.1)
        self.Dropout_02=nn.Dropout2d(0.2)

    
    def forward(self,x):
        x=self.ConvLayer1(x)
        x=self.BatchNorm1(x)
        x=self.LeakyRelU(x)
        x=self.MaxPool2d_dimhalf(x)
        x=self.Dropout_01(x)

        x=self.ConvLayer2(x)
        x=self.BatchNorm2(x)
        x=self.LeakyRelU(x)
        x=self.MaxPool2d_dimhalf(x)
        x=self.Dropout_01(x)

        x=self.ConvLayer3(x)
        x=self.BatchNorm3(x)
        x=self.LeakyRelU(x)
        x=self.MaxPool2d_dimhalf(x)
        x=self.Dropout_01(x)

        x=self.ConvLayer4(x)
        x=self.BatchNorm4(x)
        x=self.LeakyRelU(x)
        x=self.MaxPool2d_heighthalf(x)
        x=self.Dropout_02(x)

        x=self.ConvLayer5(x)
        x=self.BatchNorm5(x)
        x=self.LeakyRelU(x)
        x=self.MaxPool2d_heighthalf(x)
        x=self.Dropout_02(x)

        x=self.ConvLayer6(x)
        x=self.BatchNorm6(x)
        x=self.LeakyRelU(x)
        x=self.MaxPool2d_heighthalf(x)
        x=self.Dropout_02(x)

        return x


class v1_Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN_Encoder=CNN_Encoder()
        self.BiLSTM=nn.LSTM(input_size=256,hidden_size=256,batch_first=True,bidirectional=True,num_layers=2,dropout=0.3)
        self.Dropout=nn.Dropout(0.3)
        self.output_layer=nn.Linear(512,118)
    

    def forward(self,x):
        x=self.CNN_Encoder(x)
        x=x.squeeze(2)
        x=x.transpose(1,2)
        per_timestep_output,NONE=self.BiLSTM(x)
        LSTM_output=self.Dropout(per_timestep_output)
        logits = self.output_layer(LSTM_output)
        log_probs=nn.functional.log_softmax(logits,dim=2)
        return log_probs
    


        
