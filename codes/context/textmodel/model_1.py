import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
bertpath = '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts/bert-base-uncased'
bertlarge = '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts/bert-large-uncased'
robertapath= '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts/roberta-base'
robertalarge= '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts/roberta-large'

def validate_and_test(model, data_loader, device, num_classes):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0

        unweightet_correct = [0] * num_classes
        unweightet_total = [0] * num_classes

        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        for batch in data_loader:
            ids, net_input, net_input1, labels = batch["id"], batch["net_input"], batch["net_input1"], batch["labels"]
            text = net_input['texts_id']
            mask = net_input['texts_mask']
            text1 = net_input1['texts_id1']
            mask1 = net_input1['texts_mask1']
        #  feats1 = net_input1["feats1"]
        #  speech_padding_mask1 = net_input1["padding_mask1"]

            text = text.to(device)
            mask = mask.to(device)
            text1 = text1.to(device)
            mask1 = mask1.to(device)
            
            labels = labels.to(device)
            outputs = model(text1, mask1, text, mask)

            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
            for i in range(len(labels)):
                unweightet_total[labels[i]] += 1
                if predicted[i] == labels[i]:
                    unweightet_correct[labels[i]] += 1
                    tp[labels[i]] += 1
                else:
                    fp[predicted[i]] += 1
                    fn[labels[i]] += 1
        weighted_acc = correct / total * 100
        unweighted_acc = compute_unweighted_accuracy(unweightet_correct, unweightet_total) * 100
        weighted_f1 = compute_weighted_f1(tp, fp, fn, unweightet_total) * 100

    return weighted_acc, unweighted_acc, weighted_f1

def compute_unweighted_accuracy(list1, list2):
    result = []
    for i in range(len(list1)):
        result.append(list1[i] / list2[i])
    return sum(result)/len(result)   

def compute_weighted_f1(tp, fp, fn, unweightet_total):
    f1_scores = []
    num_classes = len(tp)
    
    for i in range(num_classes):
        if tp[i] + fp[i] == 0:
            precision = 0
        else:
            precision = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall = 0
        else:
            recall = tp[i] / (tp[i] + fn[i])
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
            
    wf1 = sum([f1_scores[i] * unweightet_total[i] for i in range(num_classes)]) / sum(unweightet_total)
    return wf1

def Attention(Q , K, V, scale):
    attention = torch.matmul(Q, K.permute(0, 2, 1))
    attention = attention * scale
    attention = F.softmax(attention, dim=-1)
    context = torch.matmul(attention, V) 
    return context
     
class BertClassifier(nn.Module):

     def __init__(self):
         super(BertClassifier, self).__init__()
         self.bert = BertModel.from_pretrained(bertpath) 
         for parameter in self.bert.parameters():
             parameter.requires_grad = False
         self.rnnA1 = nn.GRU(768, 256, 2, batch_first = True)
         self.rnnA2 = nn.GRU(256, 128, 2, batch_first = True)
         self.Q1 = nn.Linear(128, 64)
         self.K1 = nn.Linear(128, 64)
         self.V1 = nn.Linear(128, 64)
         self.actfc = nn.GELU()
         self.fc2 = nn.Linear(64 * 2, 4)
         
     def forward(self, upinput_id, upmask, input_id, mask):
         # 上文，文本1
         hidden1, pooled1 = self.bert(upinput_id, attention_mask=upmask, return_dict=False)
         # 用一个全连接层来将768维的数据映射到num_classes维上，此时也就是一个句子对应每种标签的概率
         hidden1, hnA1= self.rnnA1(hidden1)
         hnA1 = torch.transpose(hnA1, 0 ,1)
         hidden1, hnA2= self.rnnA2(hnA1)
         hnA2 = torch.transpose(hnA2, 0 ,1)
         Q1 = self.Q1(hnA2)
         K1 = self.K1(hnA2)
         V1 = self.V1(hnA2)
         dim_K1 = K1.size(-1)
         scale1 = dim_K1 ** -0.5
         context1 = Attention(Q1, K1, V1, scale1)
         out1 = context1.mean(dim=-2)
         #文本2
         hidden2, pooled2 = self.bert(input_id, attention_mask=mask, return_dict=False)
         hidden2, hnB1= self.rnnA1(hidden2)
         hnB1 = torch.transpose(hnB1, 0, 1)
         hidden2, hnB2= self.rnnA2(hnB1)
         hnB2 = torch.transpose(hnB2, 0, 1)
         Q2 = self.Q1(hnB2)
         K2 = self.K1(hnB2)
         V2 = self.V1(hnB2)
         dim_K2 = K2.size(-1)
         scale2 = dim_K2 ** -0.5
         context2 = Attention(Q2, K2, V2, scale2)
         out2 = context2.mean(dim=-2)
         out = torch.concat((out1,out2), dim= 1)
         out = self.actfc(out)
         out = self.fc2(out)
         return out