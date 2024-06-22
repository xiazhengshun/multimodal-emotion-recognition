import numpy as np

from torch.utils.data import DataLoader, random_split
import os
import contextlib
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

modelpath = '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(modelpath)
def load_dataset(data_path, encoder_path, mask_path, labels=None, min_length=3, max_length=None):
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

    return npy_data, sizes, offsets, emo_labels, encoder_data, mask_data, textsizes, textoffsets

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

        self.labels = labels
        self.texts_id = texts_id
        self.texts_mask = texts_mask
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
        
        res = {"id": index, "feats": feats, "texts_id": texts_id, "texts_mask": texts_mask}
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

        #print(np.shape(texts_id[0]))
        # texts_id = np.asarray(texts_id)
        # texts_mask = np.asarray(texts_mask)
       
        # texts_id =  texts_id.astype(np.float64)
        # texts_mask =  texts_mask.astype(np.float64)
        
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
                #"feats": collated_feats,
                #"padding_mask": padding_mask,
                "texts_id": collected_id,
                "texts_mask": collected_mask
            },
            "labels": labels
        }
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]
    
def load_ssl_features(feature_path, encoder_path, mask_path, label_dict, max_speech_seq_len=None):
    data, sizes, offsets, labels , encoder_data, mask_data, textsizes, textoffsets = load_dataset(feature_path, encoder_path, mask_path,labels='emo', min_length=1, max_length=max_speech_seq_len)
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
    
    labels = data['labels']

    test_sizes = sizes[test_start:test_end]
    test_offsets = offsets[test_start:test_end]
    test_text_sizes = textsizes[test_start:test_end]
    test_text_offsets = textoffsets[test_start:test_end]
    
    test_labels = labels[test_start:test_end]

    test_offset_start = test_offsets[0]
    test_offset_end = test_offsets[-1] + test_sizes[-1]
    test_text_offset_start = test_text_offsets[0]
    test_text_offset_end = test_text_offsets [-1] + test_text_sizes[-1]
    
    # print(test_offset_start,test_offset_end)
    # print(test_text_offset_start,test_text_offset_end)
    test_feats = feats[test_offset_start:test_offset_end, :]
    test_offsets = test_offsets - test_offset_start
    test_text_offsets = test_text_offsets - test_text_offset_start

    test_texts_id = texts_id[test_text_offset_start:test_text_offset_end, :]
    test_texts_mask = texts_mask[test_text_offset_start:test_text_offset_end, :]

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
    )

    train_val_sizes = np.concatenate([sizes[:test_start], sizes[test_end:]])
    train_val_offsets = np.concatenate([np.array([0]), np.cumsum(train_val_sizes)[:-1]], dtype=np.int64)

    train_val_text_sizes = np.concatenate([textsizes[:test_start], textsizes[test_end:]])
    train_val_text_offsets = np.concatenate([np.array([0]), np.cumsum(train_val_text_sizes)[:-1]], dtype=np.int64)
    
    train_val_labels = [item for item in labels[:test_start] + labels[test_end:]]
    
    train_val_feats = np.concatenate([feats[:test_offset_start, :], feats[test_offset_end:, :]], axis=0)
    train_val_texts_id = np.concatenate([texts_id[:test_text_offset_start, :], texts_id[test_text_offset_end:, :]], axis=0)
    train_val_texts_mask = np.concatenate([texts_mask[:test_text_offset_start, :], texts_mask[test_text_offset_end:, :]], axis=0)

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
            texts_id= train_val_texts_id,
            texts_mask= train_val_texts_mask,
            textsizes= train_val_text_sizes,
            textoffsets= train_val_text_offsets,
            labels=train_val_labels,
        )

        train_dataset, val_dataset = random_split(train_val_dataset, [train_nums, val_nums])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)
    
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, 
                            num_workers=4, pin_memory=True, shuffle=False)
    
        return train_loader, val_loader, test_loader