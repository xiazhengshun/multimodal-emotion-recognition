import torch
from torch import nn, optim
class BaseModel(nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super().__init__()
        self.pre_net = nn.Linear(input_dim, 64)

        self.post_net = nn.Linear(64 * 2, output_dim)
        
        self.activate = nn.GELU()

    def forward(self, x1, padding_mask1, x, padding_mask):
        x = self.activate(self.pre_net(x))

        x = x * (1 - padding_mask.unsqueeze(-1).float())
        x = x.sum(dim=1) / (1 - padding_mask.float()
                            ).sum(dim=1, keepdim=True)  # Compute average
        
        x1 = self.activate(self.pre_net(x1))
        x1 = x1 * (1 - padding_mask1.unsqueeze(-1).float())
        x1 = x1.sum(dim=1) / (1 - padding_mask1.float()
                            ).sum(dim=1, keepdim=True)
        x = torch.concat((x1, x), dim =1)
        x = self.post_net(x)
        return x
    
def validate_and_test(model, data_loader, device, num_classes):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0

        # unweighted accuracy
        unweightet_correct = [0] * num_classes
        unweightet_total = [0] * num_classes

        # weighted f1
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        for batch in data_loader:

            ids, net_input, net_input1, labels = batch["id"], batch["net_input"], batch["net_input1"], batch["labels"]
            feats = net_input["feats"]
            speech_padding_mask = net_input["padding_mask"]
            feats1 = net_input1["feats1"]
            speech_padding_mask1 = net_input1["padding_mask1"]

            feats = feats.to(device)
            speech_padding_mask = speech_padding_mask.to(device)
            feats1 = feats1.to(device)
            speech_padding_mask1 = speech_padding_mask1.to(device)
            
            labels = labels.to(device)

            outputs = model(feats1, speech_padding_mask1, feats, speech_padding_mask)

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