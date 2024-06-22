import numpy as np

from torch.utils.data import DataLoader, random_split
import os
import contextlib
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

modelpath = '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(modelpath)
def load_dataset(data_path, encoder_path, mask_path, data_path1, encoder_path1, mask_path1, labels=None, min_length=3, max_length=None):
    sizes = []
    offsets = []
    emo_labels = []
    textsizes = []
    textoffsets = []

    npy_data = np.load(data_path + ".npy")
    mask_data = np.load(mask_path + ".npy")
    encoder_data = np.load(encoder_path + ".npy")
    
    offset = 0
    skipped = 0
    textoffset = 0

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
                textsizes.append(1)
                textoffsets.append(textoffset)
                
                if lbl is not None:
                    emo_labels.append(lbl)
            offset += length
            textoffset += 1
            
    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)
    textsizes = np.asarray(textsizes)
    textoffsets = np.asarray(textoffsets)

    sizes1 = []
    offsets1 = []
    emo_labels1 = []
    textsizes1 = []
    textoffsets1 = []

    npy_data1 = np.load(data_path1 + ".npy")
    mask_data1 = np.load(mask_path1 + ".npy")
    encoder_data1 = np.load(encoder_path1 + ".npy")
    
    offset1 = 0
    textoffset1 = 0

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
                textsizes1.append(1)
                textoffsets1.append(textoffset1)
                
                if lbl is not None:
                    emo_labels1.append(lbl)
            offset1 += length
            textoffset1 += 1
            
    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)
    textsizes = np.asarray(textsizes)
    textoffsets = np.asarray(textoffsets)
    
    sizes1 = np.asarray(sizes1)
    offsets1 = np.asarray(offsets1)
    textsizes1 = np.asarray(textsizes1)
    textoffsets1 = np.asarray(textoffsets1)
    
    return npy_data, sizes, offsets, emo_labels, encoder_data, mask_data, textsizes, textoffsets,\
           npy_data1, sizes1, offsets1, encoder_data1, mask_data1, textsizes1, textoffsets1

class SpeechDataset(Dataset):
    def __init__(
        self,
        feats,
        sizes,
        offsets,
        texts_id,
        texts_mask,
        textsizes,
        textoffsets,
        feats1,
        sizes1,
        offsets1,
        texts_id1,
        texts_mask1,
        textsizes1,
        textoffsets1,
        labels=None,
        shuffle=True,
        sort_by_length=True,
    ):
        super().__init__()
        
        self.feats = feats
        self.sizes = sizes  # length of each sample
        self.offsets = offsets  # offset of each sample
        self.textsizes = textsizes
        self.textoffsets = textoffsets
        self.texts_id = texts_id
        self.texts_mask = texts_mask
        
        self.labels = labels
        
        self.feats1 = feats1
        self.sizes1 = sizes1  # length of each sample
        self.offsets1 = offsets1  # offset of each sample
        self.textsizes1 = textsizes1
        self.textoffsets1 = textoffsets1
        self.texts_id1 = texts_id1
        self.texts_mask1 = texts_mask1
        
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        text_offset = self.textoffsets[index]
        text_end = self.textsizes[index] + text_offset
        
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()
        texts_id = torch.from_numpy(self.texts_id[text_offset:text_end, :].copy())
        texts_mask = torch.from_numpy(self.texts_mask[text_offset:text_end, :].copy())
        
        offset1 = self.offsets1[index]
        end1 = self.sizes1[index] + offset1
     
        text_offset1 = self.textoffsets1[index]
        text_end1 = self.textsizes1[index] + text_offset1
        
        feats1 = torch.from_numpy(self.feats1[offset1:end1, :].copy()).float()
        texts_id1 = torch.from_numpy(self.texts_id1[text_offset1:text_end1, :].copy())
        texts_mask1 = torch.from_numpy(self.texts_mask1[text_offset1:text_end1, :].copy())
            
        res = {"id": index, "feats": feats, "texts_id": texts_id, "texts_mask": texts_mask, "feats1": feats1,\
                "texts_id1": texts_id1, "texts_mask1": texts_mask1}
        
        if self.labels is not None:
            res["target"] = self.labels[index]

        return res

    def __len__(self):
        return len(self.sizes)

    def collator(self, samples):
        if len(samples) == 0:
            return {}

        feats = [s["feats"] for s in samples]
        texts_id = [s["texts_id"] for s in samples]
        texts_mask = [s["texts_mask"] for s in samples]
        
        sizes = [s.shape[0] for s in feats]
        textsizes = [s.shape[0] for s in texts_id]
        labels = torch.tensor([s["target"] for s in samples]) if samples[0]["target"] is not None else None
        
        target_size = max(sizes)
        target_text_size = max(textsizes)
        
        collated_feats = feats[0].new_zeros(
             len(feats), target_size, feats[0].size(-1)
        )
        
        padding_mask = torch.BoolTensor(torch.Size([len(feats), target_size])).fill_(False)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True

        collected_id = texts_id[0].new_zeros(
            len(texts_id), 100
        )
        
        collected_mask = texts_mask[0].new_zeros(
            len(texts_mask), 100
        )
        for i, (textid, textsize) in enumerate(zip(texts_id, textsizes)):
            collected_id[i,:] = textid
        
        for i, (textmask, textsize) in enumerate(zip(texts_mask, textsizes)):    
            collected_mask[i,:] = textmask
 #history        
     
        feats1 = [s["feats1"] for s in samples]
        texts_id1 = [s["texts_id1"] for s in samples]
        texts_mask1 = [s["texts_mask1"] for s in samples]
        
        sizes1 = [s.shape[0] for s in feats1]
        
        textsizes1 = [s.shape[0] for s in texts_id1]
        
        target_size1 = max(sizes1)
        target_text_size1 = max(textsizes1)
        
        collated_feats1 = feats[0].new_zeros(
             len(feats1), target_size1, feats1[0].size(-1)
        )
        
        padding_mask1 = torch.BoolTensor(torch.Size([len(feats1), target_size1])).fill_(False)
        
        for i, (feat1, size1) in enumerate(zip(feats1, sizes1)):
            collated_feats1[i, :size1] = feat1
            padding_mask1[i, size1:] = True

        collected_id1 = texts_id1[0].new_zeros(
            len(texts_id1), 100
        )
        
        collected_mask1 = texts_mask1[0].new_zeros(
            len(texts_mask1), 100
        )
        for i, (textid1, textsize1) in enumerate(zip(texts_id1, textsizes1)):
            collected_id1[i,:] = textid1
        
        for i, (textmask1, textsize1) in enumerate(zip(texts_mask1, textsizes1)):
            collected_mask1[i,:] = textmask1
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
                "texts_id": collected_id,
                "texts_mask": collected_mask
            },
            "net_input1": {
                "feats1": collated_feats1,
                "padding_mask1": padding_mask1,
                "texts_id1": collected_id1,
                "texts_mask1": collected_mask1
            },
            "labels": labels
        }
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]
    
def load_ssl_features(feature_path, encoder_path, mask_path, feature_path1, encoder_path1, mask_path1, label_dict, max_speech_seq_len=None):
    data, sizes, offsets, labels , encoder_data, mask_data, textsizes, textoffsets, \
       data1, sizes1, offsets1, encoder_data1, mask_data1, textsizes1, textoffsets1 = \
        load_dataset(feature_path, encoder_path, mask_path, feature_path1, encoder_path1, mask_path1, labels='emo', min_length=1, max_length=max_speech_seq_len)
    labels = [ label_dict[elem] for elem in labels]
    
    num = len(labels)
    iemocap_data = {
        "feats": data,
        "sizes": sizes,
        "offsets": offsets,
        "labels": labels,
        "texts_id": encoder_data,
        "texts_mask": mask_data,
        "textsizes": textsizes,
        "textoffsets": textoffsets,
        "feats1": data1,
        "sizes1": sizes1,
        "offsets1": offsets1,
        "texts_id1": encoder_data1,
        "texts_mask1": mask_data1,
        "textsizes1": textsizes1,
        "textoffsets1": textoffsets1,
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
    texts_id, texts_mask = data['texts_id'], data['texts_mask']
    
    sizes, offsets = data['sizes'], data['offsets']
    textsizes, textoffsets = data['textsizes'], data['textoffsets']
    
    feats1 = data['feats1'] 
    texts_id1, texts_mask1 = data['texts_id1'], data['texts_mask1']
    
    sizes1, offsets1 = data['sizes1'], data['offsets1']
    textsizes1, textoffsets1 = data['textsizes1'], data['textoffsets1']
    
    labels = data['labels']

    test_sizes = sizes[test_start:test_end]
    test_offsets = offsets[test_start:test_end]
    test_text_sizes = textsizes[test_start:test_end]
    test_text_offsets = textoffsets[test_start:test_end]
    
    test_offset_start = test_offsets[0]
    test_offset_end = test_offsets[-1] + test_sizes[-1]
    test_text_offset_start = test_text_offsets[0]
    test_text_offset_end = test_text_offsets [-1] + test_text_sizes[-1]
    
    test_labels = labels[test_start:test_end]

    test_sizes1 = sizes1[test_start:test_end]
    test_offsets1 = offsets1[test_start:test_end]
    test_text_sizes1 = textsizes1[test_start:test_end]
    test_text_offsets1 = textoffsets1[test_start:test_end]
    test_offset_start1 = test_offsets1[0]
    test_offset_end1 = test_offsets1[-1] + test_sizes1[-1]
    test_text_offset_start1 = test_text_offsets1[0]
    test_text_offset_end1 = test_text_offsets1[-1] + test_text_sizes1[-1]
    
    # print(test_offset_start,test_offset_end)
    # print(test_text_offset_start,test_text_offset_end)        
    # data, 
    # batch_size,
    # test_start, 
    # test_end,
    test_feats = feats[test_offset_start:test_offset_end, :]
    test_offsets = test_offsets - test_offset_start
    test_text_offsets = test_text_offsets - test_text_offset_start

    test_texts_id = texts_id[test_text_offset_start:test_text_offset_end, :]
    test_texts_mask = texts_mask[test_text_offset_start:test_text_offset_end, :]

    test_feats1 = feats1[test_offset_start1:test_offset_end1, :]
    test_offsets1 = test_offsets1 - test_offset_start1
    test_text_offsets1 = test_text_offsets1 - test_text_offset_start1

    test_texts_id1 = texts_id1[test_text_offset_start1:test_text_offset_end1, :]
    test_texts_mask1 = texts_mask1[test_text_offset_start1:test_text_offset_end1, :]
    
    #print(np.shape(test_texts_id))
    test_dataset = SpeechDataset(
        feats=test_feats,
        sizes=test_sizes, 
        offsets=test_offsets,
        texts_id = test_texts_id,
        texts_mask = test_texts_mask,
        textsizes= test_text_sizes,
        textoffsets= test_text_offsets,
        labels=test_labels,
        feats1=test_feats1,
        sizes1=test_sizes1, 
        offsets1=test_offsets1,
        texts_id1 = test_texts_id1,
        texts_mask1 = test_texts_mask1,
        textsizes1= test_text_sizes1,
        textoffsets1= test_text_offsets1,
    )

    train_val_sizes = np.concatenate([sizes[:test_start], sizes[test_end:]])
    train_val_offsets = np.concatenate([np.array([0]), np.cumsum(train_val_sizes)[:-1]], dtype=np.int64)

    train_val_text_sizes = np.concatenate([textsizes[:test_start], textsizes[test_end:]])
    train_val_text_offsets = np.concatenate([np.array([0]), np.cumsum(train_val_text_sizes)[:-1]], dtype=np.int64)
    
    train_val_labels = [item for item in labels[:test_start] + labels[test_end:]]
    
    train_val_feats = np.concatenate([feats[:test_offset_start, :], feats[test_offset_end:, :]], axis=0)
    train_val_texts_id = np.concatenate([texts_id[:test_text_offset_start, :], texts_id[test_text_offset_end:, :]], axis=0)
    train_val_texts_mask = np.concatenate([texts_mask[:test_text_offset_start, :], texts_mask[test_text_offset_end:, :]], axis=0)

    train_val_sizes1 = np.concatenate([sizes1[:test_start], sizes1[test_end:]])
    train_val_offsets1 = np.concatenate([np.array([0]), np.cumsum(train_val_sizes1)[:-1]], dtype=np.int64)

    train_val_text_sizes1 = np.concatenate([textsizes1[:test_start], textsizes1[test_end:]])
    train_val_text_offsets1 = np.concatenate([np.array([0]), np.cumsum(train_val_text_sizes1)[:-1]], dtype=np.int64)
    
    train_val_feats1 = np.concatenate([feats1[:test_offset_start1, :], feats1[test_offset_end1:, :]], axis=0)
    train_val_texts_id1 = np.concatenate([texts_id1[:test_text_offset_start1, :], texts_id1[test_text_offset_end1:, :]], axis=0)
    train_val_texts_mask1 = np.concatenate([texts_mask1[:test_text_offset_start1, :], texts_mask1[test_text_offset_end1:, :]], axis=0)
    #print(train_val_texts_mask)
    if eval_is_test:
        train_dataset = SpeechDataset(
            feats=train_val_feats, 
            sizes=train_val_sizes, 
            offsets=train_val_offsets,
            texts_id= train_val_texts_id,
            texts_mask= train_val_texts_mask,
            textsizes= train_val_text_sizes,
            textoffsets = train_val_text_offsets,
            labels=train_val_labels,
            feats1=train_val_feats1, 
            sizes1=train_val_sizes1, 
            offsets1=train_val_offsets1,
            texts_id1= train_val_texts_id1,
            texts_mask1= train_val_texts_mask1,
            textsizes1= train_val_text_sizes1,
            textoffsets1 = train_val_text_offsets1,
        )
        
        val_dataset = test_dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)
        return train_loader, val_loader
    else:
        train_val_nums = data['num'] - (test_end - test_start)
        train_nums = int(0.8 * train_val_nums)
        val_nums = train_val_nums - train_nums

        train_val_dataset = SpeechDataset(
            feats=train_val_feats, 
            sizes=train_val_sizes, 
            offsets=train_val_offsets,
            texts_id= train_val_texts_id,
            texts_mask= train_val_texts_mask,
            textsizes= train_val_text_sizes,
            textoffsets= train_val_text_offsets,
            labels=train_val_labels,
            feats1=train_val_feats1, 
            sizes1=train_val_sizes1, 
            offsets1=train_val_offsets1,
            texts_id1= train_val_texts_id1,
            texts_mask1= train_val_texts_mask1,
            textsizes1= train_val_text_sizes1,
            textoffsets1 = train_val_text_offsets1,
        )

        train_dataset, val_dataset = random_split(train_val_dataset, [train_nums, val_nums])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)
    
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, 
                            num_workers=4, pin_memory=True, shuffle=False)
    
        return train_loader, val_loader, test_loader