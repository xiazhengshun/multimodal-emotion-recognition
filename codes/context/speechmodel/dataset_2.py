import numpy as np

from torch.utils.data import DataLoader, random_split
import os
import contextlib
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

modelpath = '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(modelpath)
def load_dataset(data_path, data_path1, data_path2, labels=None, min_length=3, max_length=None):
    sizes = []
    offsets = []
    emo_labels = []

    npy_data = np.load(data_path + ".npy")
    
    offset = 0
    skipped = 0

    if not os.path.exists(data_path + f".{labels}"):
        labels = None

    with open(data_path + ".lengths", "r") as len_f, open(
        data_path + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[
                1]  # only emo is needed
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes.append(length)
                offsets.append(offset)

                if lbl is not None:
                    emo_labels.append(lbl)
            offset += length
            
    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)

    sizes1 = []
    offsets1 = []
    emo_labels1 = []

    npy_data1 = np.load(data_path1 + ".npy")
    
    offset1 = 0
    
    if not os.path.exists(data_path1 + f".{labels}"):
        labels = None

    with open(data_path1 + ".lengths", "r") as len_f, open(
        data_path1 + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[
                1]  # only emo is needed
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes1.append(length)
                offsets1.append(offset1)
                
                if lbl is not None:
                    emo_labels1.append(lbl)
            offset1 += length

    sizes2 = []
    offsets2 = []
    emo_labels2 = []

    npy_data2 = np.load(data_path2 + ".npy")

    offset2 = 0

    if not os.path.exists(data_path2 + f".{labels}"):
        labels = None

    with open(data_path2 + ".lengths", "r") as len_f, open(
        data_path2 + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[
                1]  # only emo is needed
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes2.append(length)
                offsets2.append(offset2)
                
                if lbl is not None:
                    emo_labels2.append(lbl)
            offset2 += length
            
    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)
    
    sizes1 = np.asarray(sizes1)
    offsets1 = np.asarray(offsets1)
   
    sizes2 = np.asarray(sizes2)
    offsets2 = np.asarray(offsets2)
     
    return npy_data, sizes, offsets, emo_labels, npy_data1, sizes1, offsets1, npy_data2, sizes2, offsets2, 
 
class SpeechDataset(Dataset):
    def __init__(
        self,
        feats,
        sizes,
        offsets,
        feats1,
        sizes1,
        offsets1,
        feats2,
        sizes2,
        offsets2,
        labels=None,
        shuffle=True,
        sort_by_length=True,
    ):
        super().__init__()
        
        self.feats = feats
        self.sizes = sizes  # length of each sample
        self.offsets = offsets  # offset of each sample

        self.labels = labels

        self.feats1 = feats1
        self.sizes1 = sizes1  # length of each sample
        self.offsets1 = offsets1  # offset of each sample
        
        self.feats2 = feats2
        self.sizes2 = sizes2  # length of each sample
        self.offsets2 = offsets2  # offset of each sample

        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset     
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()
                
        offset1 = self.offsets1[index]
        end1 = self.sizes1[index] + offset1    
        feats1 = torch.from_numpy(self.feats1[offset1:end1, :].copy()).float()
          
        offset2 = self.offsets2[index]
        end2 = self.sizes2[index] + offset2
        feats2 = torch.from_numpy(self.feats2[offset2:end2, :].copy()).float()
            
        res = {"id": index, "feats": feats, "feats1": feats1, "feats2": feats2}
        
        if self.labels is not None:
            res["target"] = self.labels[index]

        return res

    def __len__(self):
        return len(self.sizes)

    def collator(self, samples):
        if len(samples) == 0:
            return {}

        feats = [s["feats"] for s in samples]
        sizes = [s.shape[0] for s in feats]
        labels = torch.tensor([s["target"] for s in samples]) if samples[0]["target"] is not None else None
        
        target_size = max(sizes)
        
        collated_feats = feats[0].new_zeros(
             len(feats), target_size, feats[0].size(-1)
        )
        
        padding_mask = torch.BoolTensor(torch.Size([len(feats), target_size])).fill_(False)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True

 #history        
        feats1 = [s["feats1"] for s in samples]
        sizes1 = [s.shape[0] for s in feats1]  
        target_size1 = max(sizes1)
        
        collated_feats1 = feats[0].new_zeros(
             len(feats1), target_size1, feats1[0].size(-1)
        )
        
        padding_mask1 = torch.BoolTensor(torch.Size([len(feats1), target_size1])).fill_(False)
        
        for i, (feat1, size1) in enumerate(zip(feats1, sizes1)):
            collated_feats1[i, :size1] = feat1
            padding_mask1[i, size1:] = True
            
#feats 2
        feats2 = [s["feats2"] for s in samples]
        sizes2 = [s.shape[0] for s in feats2]
        target_size2 = max(sizes2)
        
        collated_feats2 = feats[0].new_zeros(
             len(feats2), target_size2, feats2[0].size(-1)
        )
        
        padding_mask2 = torch.BoolTensor(torch.Size([len(feats2), target_size2])).fill_(False)
        
        for i, (feat2, size2) in enumerate(zip(feats2, sizes2)):
            collated_feats2[i, :size2] = feat2
            padding_mask2[i, size2:] = True
        # collated_feats = feats[0].new_zeros(
        #      len(feats), target_size, feats[0].size(-1)
        # )
        # padding_mask = torch.BoolTensor(torch.Size([len(feats), target_size])).fill_(False)
        # for i, (feat, size) in enumerate(zip(feats, sizes)):
        #     collated_feats[i, :size] = feat
        #     padding_mask[i, size:] = True
        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "feats": collated_feats,
                "padding_mask": padding_mask,
            },
            "net_input1": {
                "feats1": collated_feats1,
                "padding_mask1": padding_mask1,
            },
            "net_input2": {
                "feats2": collated_feats2,
                "padding_mask2": padding_mask2,
            },
            "labels": labels
        }
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]
    
def load_ssl_features(feature_path, feature_path1, feature_path2, label_dict, max_speech_seq_len=None):
    data, sizes, offsets, labels , data1, sizes1, offsets1, data2, sizes2, offsets2 = \
        load_dataset(feature_path, feature_path1, feature_path2, labels='emo', min_length=1, max_length=max_speech_seq_len)
    labels = [ label_dict[elem] for elem in labels]
    
    num = len(labels)
    iemocap_data = {
        "feats": data,
        "sizes": sizes,
        "offsets": offsets,
        "labels": labels,
        "feats1": data1,
        "sizes1": sizes1,
        "offsets1": offsets1,
        "feats2": data2,
        "sizes2": sizes2,
        "offsets2": offsets2,
        "num": num
    } 

    return iemocap_data


def train_valid_test_iemocap_dataloader(
        data, 
        batch_size,
        test_start, 
        test_end,
        eval_is_test=True,
    ):
    feats = data['feats']     
    sizes, offsets = data['sizes'], data['offsets']
    
    feats1 = data['feats1'] 
    sizes1, offsets1 = data['sizes1'], data['offsets1']
    
    feats2 = data['feats2']    
    sizes2, offsets2 = data['sizes2'], data['offsets2']
    
    labels = data['labels']

    test_sizes = sizes[test_start:test_end]
    test_offsets = offsets[test_start:test_end]
    test_offset_start = test_offsets[0]
    test_offset_end = test_offsets[-1] + test_sizes[-1]

    test_labels = labels[test_start:test_end]

    test_sizes1 = sizes1[test_start:test_end]
    test_offsets1 = offsets1[test_start:test_end]
    test_offset_start1 = test_offsets1[0]
    test_offset_end1 = test_offsets1[-1] + test_sizes1[-1]
    
    test_sizes2 = sizes2[test_start:test_end]
    test_offsets2 = offsets2[test_start:test_end]
    test_offset_start2 = test_offsets2[0]
    test_offset_end2 = test_offsets2[-1] + test_sizes2[-1]
    
    test_feats = feats[test_offset_start:test_offset_end, :]
    test_offsets = test_offsets - test_offset_start

    test_feats1 = feats1[test_offset_start1:test_offset_end1, :]
    test_offsets1 = test_offsets1 - test_offset_start1

    test_feats2 = feats2[test_offset_start2:test_offset_end2, :]
    test_offsets2 = test_offsets2 - test_offset_start2
       
    #print(np.shape(test_texts_id))
    test_dataset = SpeechDataset(
        feats=test_feats,
        sizes=test_sizes, 
        offsets=test_offsets,
        labels=test_labels,
        feats1=test_feats1,
        sizes1=test_sizes1, 
        offsets1=test_offsets1,
        feats2=test_feats2,
        sizes2=test_sizes2, 
        offsets2=test_offsets2,
    )

    train_val_sizes = np.concatenate([sizes[:test_start], sizes[test_end:]])
    train_val_offsets = np.concatenate([np.array([0]), np.cumsum(train_val_sizes)[:-1]], dtype=np.int64)

    train_val_feats = np.concatenate([feats[:test_offset_start, :], feats[test_offset_end:, :]], axis=0)
    
    train_val_labels = [item for item in labels[:test_start] + labels[test_end:]]
    
    train_val_sizes1 = np.concatenate([sizes1[:test_start], sizes1[test_end:]])
    train_val_offsets1 = np.concatenate([np.array([0]), np.cumsum(train_val_sizes1)[:-1]], dtype=np.int64)
    
    train_val_feats1 = np.concatenate([feats1[:test_offset_start1, :], feats1[test_offset_end1:, :]], axis=0)
    
    train_val_sizes2 = np.concatenate([sizes2[:test_start], sizes2[test_end:]])
    train_val_offsets2 = np.concatenate([np.array([0]), np.cumsum(train_val_sizes2)[:-1]], dtype=np.int64)
    
    train_val_feats2 = np.concatenate([feats2[:test_offset_start2, :], feats2[test_offset_end2:, :]], axis=0)

    
    #print(train_val_texts_mask)
    if eval_is_test:
        train_dataset = SpeechDataset(
            feats=train_val_feats, 
            sizes=train_val_sizes, 
            offsets=train_val_offsets,
            labels=train_val_labels,
            feats1=train_val_feats1, 
            sizes1=train_val_sizes1, 
            offsets1=train_val_offsets1,
            feats2=train_val_feats2, 
            sizes2=train_val_sizes2, 
            offsets2=train_val_offsets2
        )
        
        val_dataset = test_dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)
        return train_loader, val_loader, val_loader
    else:
        train_val_nums = data['num'] - (test_end - test_start)
        train_nums = int(0.8 * train_val_nums)
        val_nums = train_val_nums - train_nums

        train_val_dataset = SpeechDataset(
            feats=train_val_feats, 
            sizes=train_val_sizes, 
            offsets=train_val_offsets,
            labels=train_val_labels,
            feats1=train_val_feats1, 
            sizes1=train_val_sizes1, 
            offsets1=train_val_offsets1,
            feats2=train_val_feats2, 
            sizes2=train_val_sizes2, 
            offsets2=train_val_offsets2
        )

        train_dataset, val_dataset = random_split(train_val_dataset, [train_nums, val_nums])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)
    
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, 
                            num_workers=4, pin_memory=True, shuffle=False)
    
        return train_loader, val_loader, test_loader