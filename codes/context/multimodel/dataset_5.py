import numpy as np

from torch.utils.data import DataLoader, random_split
import os
import contextlib
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

modelpath = '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(modelpath)
def load_dataset(data_path, encoder_path, mask_path, data_path1, encoder_path1, mask_path1, data_path2, \
                 encoder_path2, mask_path2, data_path3, encoder_path3, mask_path3, data_path4, encoder_path4,\
                 mask_path4, data_path5, encoder_path5, mask_path5, labels=None, min_length=3, max_length=None):
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

    sizes2 = []
    offsets2 = []
    emo_labels2 = []
    textsizes2 = []
    textoffsets2 = []

    npy_data2 = np.load(data_path2 + ".npy")
    mask_data2 = np.load(mask_path2 + ".npy")
    encoder_data2 = np.load(encoder_path2 + ".npy")
    
    offset2 = 0
    textoffset2 = 0

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
                textsizes2.append(1)
                textoffsets2.append(textoffset2)
                
                if lbl is not None:
                    emo_labels2.append(lbl)
            offset2 += length
            textoffset2 += 1
            
    sizes3 = []
    offsets3 = []
    emo_labels3 = []
    textsizes3 = []
    textoffsets3 = []

    npy_data3 = np.load(data_path3 + ".npy")
    mask_data3 = np.load(mask_path3 + ".npy")
    encoder_data3 = np.load(encoder_path3 + ".npy")
    
    offset3 = 0
    textoffset3 = 0

    if not os.path.exists(data_path3 + f".{labels}"):
        labels = None

    with open(data_path3 + ".lengths", "r") as len_f, open(
        data_path3 + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[
                1]  # only emo is needed
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes3.append(length)
                offsets3.append(offset3)
                textsizes3.append(1)
                textoffsets3.append(textoffset3)
                
                if lbl is not None:
                    emo_labels3.append(lbl)
            offset3 += length
            textoffset3 += 1
            
    sizes4 = []
    offsets4 = []
    emo_labels4 = []
    textsizes4 = []
    textoffsets4 = []

    npy_data4 = np.load(data_path4 + ".npy")
    mask_data4 = np.load(mask_path4 + ".npy")
    encoder_data4 = np.load(encoder_path4 + ".npy")
    
    offset4 = 0
    textoffset4 = 0

    if not os.path.exists(data_path4 + f".{labels}"):
        labels = None

    with open(data_path4 + ".lengths", "r") as len_f, open(
        data_path4 + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[
                1]  # only emo is needed
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes4.append(length)
                offsets4.append(offset4)
                textsizes4.append(1)
                textoffsets4.append(textoffset4)
                
                if lbl is not None:
                    emo_labels4.append(lbl)
            offset4 += length
            textoffset4 += 1
     
    sizes5 = []
    offsets5 = []
    emo_labels5 = []
    textsizes5 = []
    textoffsets5 = []

    npy_data5 = np.load(data_path5 + ".npy")
    mask_data5 = np.load(mask_path5 + ".npy")
    encoder_data5 = np.load(encoder_path5 + ".npy")
    
    offset5 = 0
    textoffset5 = 0

    if not os.path.exists(data_path5 + f".{labels}"):
        labels = None

    with open(data_path5 + ".lengths", "r") as len_f, open(
        data_path5 + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[
                1]  # only emo is needed
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes5.append(length)
                offsets5.append(offset5)
                textsizes5.append(1)
                textoffsets5.append(textoffset5)
                
                if lbl is not None:
                    emo_labels5.append(lbl)
            offset5 += length
            textoffset5 += 1
                   
    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)
    textsizes = np.asarray(textsizes)
    textoffsets = np.asarray(textoffsets)
    
    sizes1 = np.asarray(sizes1)
    offsets1 = np.asarray(offsets1)
    textsizes1 = np.asarray(textsizes1)
    textoffsets1 = np.asarray(textoffsets1)
   
    sizes2 = np.asarray(sizes2)
    offsets2 = np.asarray(offsets2)
    textsizes2 = np.asarray(textsizes2)
    textoffsets2 = np.asarray(textoffsets2)
     
    sizes3 = np.asarray(sizes3)
    offsets3 = np.asarray(offsets3)
    textsizes3 = np.asarray(textsizes3)
    textoffsets3 = np.asarray(textoffsets3)
    
    sizes4 = np.asarray(sizes4)
    offsets4 = np.asarray(offsets4)
    textsizes4 = np.asarray(textsizes4)
    textoffsets4 = np.asarray(textoffsets4)
    
    sizes5 = np.asarray(sizes5)
    offsets5 = np.asarray(offsets5)
    textsizes5 = np.asarray(textsizes5)
    textoffsets5 = np.asarray(textoffsets5)
    
    return  npy_data, sizes, offsets, emo_labels, encoder_data, mask_data, textsizes, textoffsets,\
            npy_data1, sizes1, offsets1, encoder_data1, mask_data1, textsizes1, textoffsets1, \
            npy_data2, sizes2, offsets2, encoder_data2, mask_data2, textsizes2, textoffsets2, \
            npy_data3, sizes3, offsets3, encoder_data3, mask_data3, textsizes3, textoffsets3, \
            npy_data4, sizes4, offsets4, encoder_data4, mask_data4, textsizes4, textoffsets4, \
            npy_data5, sizes5, offsets5, encoder_data5, mask_data5, textsizes5, textoffsets5
 
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
        feats2,
        sizes2,
        offsets2,
        texts_id2,
        texts_mask2,
        textsizes2,
        textoffsets2,
        feats3,
        sizes3,
        offsets3,
        texts_id3,
        texts_mask3,
        textsizes3,
        textoffsets3,
        feats4,
        sizes4,
        offsets4,
        texts_id4,
        texts_mask4,
        textsizes4,
        textoffsets4,
        feats5,
        sizes5,
        offsets5,
        texts_id5,
        texts_mask5,
        textsizes5,
        textoffsets5,
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
        
        self.feats1 = feats1
        self.sizes1 = sizes1  # length of each sample
        self.offsets1 = offsets1  # offset of each sample
        self.textsizes1 = textsizes1
        self.textoffsets1 = textoffsets1
        self.texts_id1 = texts_id1
        self.texts_mask1 = texts_mask1

        self.feats2 = feats2
        self.sizes2 = sizes2  # length of each sample
        self.offsets2 = offsets2  # offset of each sample
        self.textsizes2 = textsizes2
        self.textoffsets2 = textoffsets2
        self.texts_id2 = texts_id2
        self.texts_mask2 = texts_mask2

        self.feats3 = feats3
        self.sizes3 = sizes3  # length of each sample
        self.offsets3 = offsets3  # offset of each sample
        self.textsizes3 = textsizes3
        self.textoffsets3 = textoffsets3
        self.texts_id3 = texts_id3
        self.texts_mask3 = texts_mask3
        
        self.feats4 = feats4
        self.sizes4 = sizes4  # length of each sample
        self.offsets4 = offsets4  # offset of each sample
        self.textsizes4 = textsizes4
        self.textoffsets4 = textoffsets4
        self.texts_id4 = texts_id4
        self.texts_mask4 = texts_mask4
        
        self.feats5 = feats5
        self.sizes5 = sizes5  # length of each sample
        self.offsets5 = offsets5  # offset of each sample
        self.textsizes5 = textsizes5
        self.textoffsets5 = textoffsets5
        self.texts_id5 = texts_id5
        self.texts_mask5 = texts_mask5
        
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
        feats1 = torch.from_numpy(self.feats1[offset1:end1, :].copy()).float()   
            
        text_offset1 = self.textoffsets1[index]
        text_end1 = self.textsizes1[index] + text_offset1
        texts_id1 = torch.from_numpy(self.texts_id1[text_offset1:text_end1, :].copy())
        texts_mask1 = torch.from_numpy(self.texts_mask1[text_offset1:text_end1, :].copy())
            
        offset2 = self.offsets2[index]
        end2 = self.sizes2[index] + offset2
        feats2 = torch.from_numpy(self.feats2[offset2:end2, :].copy()).float()   
            
        text_offset2 = self.textoffsets2[index]
        text_end2 = self.textsizes2[index] + text_offset2
        texts_id2 = torch.from_numpy(self.texts_id2[text_offset2:text_end2, :].copy())
        texts_mask2 = torch.from_numpy(self.texts_mask2[text_offset2:text_end2, :].copy())
            
        offset3 = self.offsets3[index]
        end3 = self.sizes3[index] + offset3
        feats3 = torch.from_numpy(self.feats3[offset3:end3, :].copy()).float()   
            
        text_offset3 = self.textoffsets3[index]
        text_end3 = self.textsizes3[index] + text_offset3
        texts_id3 = torch.from_numpy(self.texts_id3[text_offset3:text_end3, :].copy())
        texts_mask3 = torch.from_numpy(self.texts_mask3[text_offset3:text_end3, :].copy())
            
        offset4 = self.offsets4[index]
        end4 = self.sizes4[index] + offset4
        feats4 = torch.from_numpy(self.feats4[offset4:end4, :].copy()).float()   
            
        text_offset4 = self.textoffsets4[index]
        text_end4 = self.textsizes4[index] + text_offset4
        texts_id4 = torch.from_numpy(self.texts_id4[text_offset4:text_end4, :].copy())
        texts_mask4 = torch.from_numpy(self.texts_mask4[text_offset4:text_end4, :].copy())
            
        offset5 = self.offsets5[index]
        end5 = self.sizes5[index] + offset5
        feats5 = torch.from_numpy(self.feats5[offset5:end5, :].copy()).float()   
            
        text_offset5 = self.textoffsets5[index]
        text_end5 = self.textsizes5[index] + text_offset5
        texts_id5 = torch.from_numpy(self.texts_id5[text_offset5:text_end5, :].copy())
        texts_mask5 = torch.from_numpy(self.texts_mask5[text_offset5:text_end5, :].copy())
            
        res = {"id": index, "feats": feats, "texts_id": texts_id, "texts_mask": texts_mask, "feats1": feats1,\
                "texts_id1": texts_id1, "texts_mask1": texts_mask1, "feats2": feats2, "texts_id2": texts_id2, \
                "texts_mask2": texts_mask2, "feats3": feats3, "texts_id3": texts_id3, "texts_mask3": texts_mask3,\
                "feats4": feats4, "texts_id4": texts_id4, "texts_mask4": texts_mask4, "feats5": feats5, "texts_id5": texts_id5, "texts_mask5": texts_mask5}
        
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
            
#feats 2
        feats2 = [s["feats2"] for s in samples]
        texts_id2 = [s["texts_id2"] for s in samples]
        texts_mask2 = [s["texts_mask2"] for s in samples]
        
        sizes2 = [s.shape[0] for s in feats2]
        
        textsizes2 = [s.shape[0] for s in texts_id2]
        
        target_size2 = max(sizes2)
        target_text_size2 = max(textsizes2)
        
        collated_feats2 = feats[0].new_zeros(
             len(feats2), target_size2, feats2[0].size(-1)
        )
        
        padding_mask2 = torch.BoolTensor(torch.Size([len(feats2), target_size2])).fill_(False)
        
        for i, (feat2, size2) in enumerate(zip(feats2, sizes2)):
            collated_feats2[i, :size2] = feat2
            padding_mask2[i, size2:] = True

        collected_id2 = texts_id2[0].new_zeros(
            len(texts_id2), 100
        )
        
        collected_mask2 = texts_mask2[0].new_zeros(
            len(texts_mask2), 100
        )
        for i, (textid2, textsize1) in enumerate(zip(texts_id2, textsizes2)):
            collected_id2[i,:] = textid2
        
        for i, (textmask2, textsize1) in enumerate(zip(texts_mask2, textsizes2)):
            collected_mask2[i,:] = textmask2
            
#feats 3
        feats3 = [s["feats3"] for s in samples]
        texts_id3 = [s["texts_id3"] for s in samples]
        texts_mask3 = [s["texts_mask3"] for s in samples]
        
        sizes3 = [s.shape[0] for s in feats3]
        
        textsizes3 = [s.shape[0] for s in texts_id3]
        
        target_size3 = max(sizes3)
        target_text_size3 = max(textsizes3)
        
        collated_feats3 = feats[0].new_zeros(
             len(feats3), target_size3, feats3[0].size(-1)
        )
        
        padding_mask3 = torch.BoolTensor(torch.Size([len(feats3), target_size3])).fill_(False)
        
        for i, (feat3, size3) in enumerate(zip(feats3, sizes3)):
            collated_feats3[i, :size3] = feat3
            padding_mask3[i, size3:] = True

        collected_id3 = texts_id3[0].new_zeros(
            len(texts_id3), 100
        )
        
        collected_mask3 = texts_mask3[0].new_zeros(
            len(texts_mask3), 100
        )
        for i, (textid3, textsize1) in enumerate(zip(texts_id3, textsizes3)):
            collected_id3[i,:] = textid3
        
        for i, (textmask3, textsize1) in enumerate(zip(texts_mask3, textsizes3)):
            collected_mask3[i,:] = textmask3
            
#feats 4
        feats4 = [s["feats4"] for s in samples]
        texts_id4 = [s["texts_id4"] for s in samples]
        texts_mask4 = [s["texts_mask4"] for s in samples]
        
        sizes4 = [s.shape[0] for s in feats4]
        
        textsizes4 = [s.shape[0] for s in texts_id4]
        
        target_size4 = max(sizes4)
        target_text_size4 = max(textsizes4)
        
        collated_feats4 = feats4[0].new_zeros(
             len(feats4), target_size4, feats4[0].size(-1)
        )
        
        padding_mask4 = torch.BoolTensor(torch.Size([len(feats4), target_size4])).fill_(False)
        
        for i, (feat4, size4) in enumerate(zip(feats4, sizes4)):
            collated_feats4[i, :size4] = feat4
            padding_mask4[i, size4:] = True

        collected_id4 = texts_id4[0].new_zeros(
            len(texts_id4), 100
        )
        
        collected_mask4 = texts_mask4[0].new_zeros(
            len(texts_mask4), 100
        )
        for i, (textid4, textsize1) in enumerate(zip(texts_id4, textsizes4)):
            collected_id4[i,:] = textid4
        
        for i, (textmask4, textsize1) in enumerate(zip(texts_mask4, textsizes4)):
            collected_mask4[i,:] = textmask4
            
#feats 5
        feats5 = [s["feats5"] for s in samples]
        texts_id5 = [s["texts_id5"] for s in samples]
        texts_mask5 = [s["texts_mask5"] for s in samples]
        
        sizes5 = [s.shape[0] for s in feats5]
        
        textsizes5 = [s.shape[0] for s in texts_id5]
        
        target_size5 = max(sizes5)
        target_text_size5 = max(textsizes5)
        
        collated_feats5 = feats5[0].new_zeros(
             len(feats5), target_size5, feats5[0].size(-1)
        )
        
        padding_mask5 = torch.BoolTensor(torch.Size([len(feats5), target_size5])).fill_(False)
        
        for i, (feat5, size5) in enumerate(zip(feats5, sizes5)):
            collated_feats5[i, :size5] = feat5
            padding_mask5[i, size5:] = True

        collected_id5 = texts_id5[0].new_zeros(
            len(texts_id5), 100
        )
        
        collected_mask5 = texts_mask5[0].new_zeros(
            len(texts_mask5), 100
        )
        for i, (textid5, textsize5) in enumerate(zip(texts_id5, textsizes5)):
            collected_id5[i,:] = textid5
        
        for i, (textmask5, textsize5) in enumerate(zip(texts_mask5, textsizes5)):
            collected_mask5[i,:] = textmask5

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
            "net_input2": {
                "feats2": collated_feats2,
                "padding_mask2": padding_mask2,
                "texts_id2": collected_id2,
                "texts_mask2": collected_mask2
            },
            "net_input3": {
                "feats3": collated_feats3,
                "padding_mask3": padding_mask3,
                "texts_id3": collected_id3,
                "texts_mask3": collected_mask3
            },
            "net_input4": {
                "feats4": collated_feats4,
                "padding_mask4": padding_mask4,
                "texts_id4": collected_id4,
                "texts_mask4": collected_mask4
            },
            "net_input5": {
                "feats5": collated_feats5,
                "padding_mask5": padding_mask5,
                "texts_id5": collected_id5,
                "texts_mask5": collected_mask5
            },
            "labels": labels
        }
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]
    
def load_ssl_features(feature_path, encoder_path, mask_path, feature_path1, encoder_path1, mask_path1, feature_path2,\
                  encoder_path2, mask_path2, feature_path3, encoder_path3, mask_path3, feature_path4, encoder_path4, \
                  mask_path4, feature_path5, encoder_path5, mask_path5, label_dict, max_speech_seq_len=None):
    data, sizes, offsets, labels , encoder_data, mask_data, textsizes, textoffsets, \
       data1, sizes1, offsets1, encoder_data1, mask_data1, textsizes1, textoffsets1,\
        data2, sizes2, offsets2, encoder_data2, mask_data2, textsizes2, textoffsets2,\
         data3, sizes3, offsets3, encoder_data3, mask_data3, textsizes3, textoffsets3,\
          data4, sizes4, offsets4, encoder_data4, mask_data4, textsizes4, textoffsets4, \
           data5, sizes5, offsets5, encoder_data5, mask_data5, textsizes5, textoffsets5 = \
        load_dataset(feature_path, encoder_path, mask_path, feature_path1, encoder_path1, mask_path1, feature_path2, \
                    encoder_path2, mask_path2, feature_path3, encoder_path3, mask_path3, feature_path4, encoder_path4,\
                    mask_path4, feature_path5, encoder_path5, mask_path5, labels='emo', min_length=1, max_length=max_speech_seq_len)
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
        "feats2": data2,
        "sizes2": sizes2,
        "offsets2": offsets2,
        "texts_id2": encoder_data2,
        "texts_mask2": mask_data2,
        "textsizes2": textsizes2,
        "textoffsets2": textoffsets2,
        "feats3": data3,
        "sizes3": sizes3,
        "offsets3": offsets3,
        "texts_id3": encoder_data3,
        "texts_mask3": mask_data3,
        "textsizes3": textsizes3,
        "textoffsets3": textoffsets3,
        "feats4": data4,
        "sizes4": sizes4,
        "offsets4": offsets4,
        "texts_id4": encoder_data4,
        "texts_mask4": mask_data4,
        "textsizes4": textsizes4,
        "textoffsets4": textoffsets4,
        "feats5": data5,
        "sizes5": sizes5,
        "offsets5": offsets5,
        "texts_id5": encoder_data5,
        "texts_mask5": mask_data5,
        "textsizes5": textsizes5,
        "textoffsets5": textoffsets5,
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
    
    feats2 = data['feats2'] 
    texts_id2, texts_mask2 = data['texts_id2'], data['texts_mask2']
    
    sizes2, offsets2 = data['sizes2'], data['offsets2']
    textsizes2, textoffsets2 = data['textsizes2'], data['textoffsets2']
    
    feats3 = data['feats3'] 
    texts_id3, texts_mask3 = data['texts_id3'], data['texts_mask3']
    
    sizes3, offsets3 = data['sizes3'], data['offsets3']
    textsizes3, textoffsets3 = data['textsizes3'], data['textoffsets3']
    
    feats4 = data['feats4'] 
    texts_id4, texts_mask4 = data['texts_id4'], data['texts_mask4']
    
    sizes4, offsets4 = data['sizes4'], data['offsets4']
    textsizes4, textoffsets4 = data['textsizes4'], data['textoffsets4']
    
    feats5 = data['feats5'] 
    texts_id5, texts_mask5 = data['texts_id5'], data['texts_mask5']
    
    sizes5, offsets5 = data['sizes5'], data['offsets5']
    textsizes5, textoffsets5 = data['textsizes5'], data['textoffsets5']
    
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
    
    test_sizes2 = sizes2[test_start:test_end]
    test_offsets2 = offsets2[test_start:test_end]
    test_text_sizes2 = textsizes2[test_start:test_end]
    test_text_offsets2 = textoffsets2[test_start:test_end]
    test_offset_start2 = test_offsets2[0]
    test_offset_end2 = test_offsets2[-1] + test_sizes2[-1]
    test_text_offset_start2 = test_text_offsets2[0]
    test_text_offset_end2 = test_text_offsets2[-1] + test_text_sizes2[-1]
    
    test_sizes3 = sizes3[test_start:test_end]
    test_offsets3 = offsets3[test_start:test_end]
    test_text_sizes3 = textsizes3[test_start:test_end]
    test_text_offsets3 = textoffsets3[test_start:test_end]
    test_offset_start3 = test_offsets3[0]
    test_offset_end3 = test_offsets3[-1] + test_sizes3[-1]
    test_text_offset_start3 = test_text_offsets3[0]
    test_text_offset_end3 = test_text_offsets3[-1] + test_text_sizes3[-1]
    
    test_sizes4 = sizes4[test_start:test_end]
    test_offsets4 = offsets4[test_start:test_end]
    test_text_sizes4 = textsizes4[test_start:test_end]
    test_text_offsets4 = textoffsets4[test_start:test_end]
    test_offset_start4 = test_offsets4[0]
    test_offset_end4 = test_offsets4[-1] + test_sizes4[-1]
    test_text_offset_start4 = test_text_offsets4[0]
    test_text_offset_end4 = test_text_offsets4[-1] + test_text_sizes4[-1]

    test_sizes5 = sizes5[test_start:test_end]
    test_offsets5 = offsets5[test_start:test_end]
    test_text_sizes5 = textsizes5[test_start:test_end]
    test_text_offsets5 = textoffsets5[test_start:test_end]
    test_offset_start5 = test_offsets5[0]
    test_offset_end5 = test_offsets5[-1] + test_sizes5[-1]
    test_text_offset_start5 = test_text_offsets5[0]
    test_text_offset_end5 = test_text_offsets5[-1] + test_text_sizes5[-1]
    
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

    test_feats2 = feats2[test_offset_start2:test_offset_end2, :]
    test_offsets2 = test_offsets2 - test_offset_start2
    test_text_offsets2 = test_text_offsets2 - test_text_offset_start2

    test_texts_id2 = texts_id2[test_text_offset_start2:test_text_offset_end2, :]
    test_texts_mask2 = texts_mask2[test_text_offset_start2:test_text_offset_end2, :]
       
    test_feats3 = feats3[test_offset_start3:test_offset_end3, :]
    test_offsets3 = test_offsets3 - test_offset_start3
    test_text_offsets3 = test_text_offsets3 - test_text_offset_start3

    test_texts_id3 = texts_id3[test_text_offset_start3:test_text_offset_end3, :]
    test_texts_mask3 = texts_mask3[test_text_offset_start3:test_text_offset_end3, :]
    
    test_feats4 = feats4[test_offset_start4:test_offset_end4, :]
    test_offsets4 = test_offsets4 - test_offset_start4
    test_text_offsets4 = test_text_offsets4 - test_text_offset_start4

    test_texts_id4 = texts_id4[test_text_offset_start4:test_text_offset_end4, :]
    test_texts_mask4 = texts_mask4[test_text_offset_start4:test_text_offset_end4, :]
    
    test_feats5 = feats5[test_offset_start5:test_offset_end5, :]
    test_offsets5 = test_offsets5 - test_offset_start5
    test_text_offsets5 = test_text_offsets5 - test_text_offset_start5

    test_texts_id5 = texts_id5[test_text_offset_start5:test_text_offset_end5, :]
    test_texts_mask5 = texts_mask5[test_text_offset_start5:test_text_offset_end5, :]
    
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
        feats2=test_feats2,
        sizes2=test_sizes2, 
        offsets2=test_offsets2,
        texts_id2 = test_texts_id2,
        texts_mask2 = test_texts_mask2,
        textsizes2= test_text_sizes2,
        textoffsets2= test_text_offsets2,
        feats3=test_feats3,
        sizes3=test_sizes3, 
        offsets3=test_offsets3,
        texts_id3 = test_texts_id3,
        texts_mask3 = test_texts_mask3,
        textsizes3= test_text_sizes3,
        textoffsets3= test_text_offsets3,
        feats4=test_feats4,
        sizes4=test_sizes4, 
        offsets4=test_offsets4,
        texts_id4 = test_texts_id4,
        texts_mask4 = test_texts_mask4,
        textsizes4= test_text_sizes4,
        textoffsets4= test_text_offsets4,
        feats5=test_feats5,
        sizes5=test_sizes5, 
        offsets5=test_offsets5,
        texts_id5 = test_texts_id5,
        texts_mask5 = test_texts_mask5,
        textsizes5= test_text_sizes5,
        textoffsets5= test_text_offsets5
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
    
    train_val_sizes2 = np.concatenate([sizes2[:test_start], sizes2[test_end:]])
    train_val_offsets2 = np.concatenate([np.array([0]), np.cumsum(train_val_sizes2)[:-1]], dtype=np.int64)

    train_val_text_sizes2 = np.concatenate([textsizes2[:test_start], textsizes2[test_end:]])
    train_val_text_offsets2 = np.concatenate([np.array([0]), np.cumsum(train_val_text_sizes2)[:-1]], dtype=np.int64)
    
    train_val_feats2 = np.concatenate([feats2[:test_offset_start2, :], feats2[test_offset_end2:, :]], axis=0)
    train_val_texts_id2 = np.concatenate([texts_id2[:test_text_offset_start2, :], texts_id2[test_text_offset_end2:, :]], axis=0)
    train_val_texts_mask2 = np.concatenate([texts_mask2[:test_text_offset_start2, :], texts_mask2[test_text_offset_end2:, :]], axis=0)
   
    train_val_sizes3 = np.concatenate([sizes3[:test_start], sizes3[test_end:]])
    train_val_offsets3 = np.concatenate([np.array([0]), np.cumsum(train_val_sizes3)[:-1]], dtype=np.int64)

    train_val_text_sizes3 = np.concatenate([textsizes3[:test_start], textsizes3[test_end:]])
    train_val_text_offsets3 = np.concatenate([np.array([0]), np.cumsum(train_val_text_sizes3)[:-1]], dtype=np.int64)
    
    train_val_feats3 = np.concatenate([feats3[:test_offset_start3, :], feats3[test_offset_end3:, :]], axis=0)
    train_val_texts_id3 = np.concatenate([texts_id3[:test_text_offset_start3, :], texts_id3[test_text_offset_end3:, :]], axis=0)
    train_val_texts_mask3 = np.concatenate([texts_mask3[:test_text_offset_start3, :], texts_mask3[test_text_offset_end3:, :]], axis=0)
     
    train_val_sizes4 = np.concatenate([sizes4[:test_start], sizes4[test_end:]])
    train_val_offsets4 = np.concatenate([np.array([0]), np.cumsum(train_val_sizes4)[:-1]], dtype=np.int64)

    train_val_text_sizes4 = np.concatenate([textsizes4[:test_start], textsizes4[test_end:]])
    train_val_text_offsets4 = np.concatenate([np.array([0]), np.cumsum(train_val_text_sizes4)[:-1]], dtype=np.int64)
    
    train_val_feats4 = np.concatenate([feats4[:test_offset_start4, :], feats4[test_offset_end4:, :]], axis=0)
    train_val_texts_id4 = np.concatenate([texts_id4[:test_text_offset_start4, :], texts_id4[test_text_offset_end4:, :]], axis=0)
    train_val_texts_mask4 = np.concatenate([texts_mask4[:test_text_offset_start4, :], texts_mask4[test_text_offset_end4:, :]], axis=0)
    
    train_val_sizes5 = np.concatenate([sizes5[:test_start], sizes5[test_end:]])
    train_val_offsets5 = np.concatenate([np.array([0]), np.cumsum(train_val_sizes5)[:-1]], dtype=np.int64)

    train_val_text_sizes5 = np.concatenate([textsizes5[:test_start], textsizes5[test_end:]])
    train_val_text_offsets5 = np.concatenate([np.array([0]), np.cumsum(train_val_text_sizes5)[:-1]], dtype=np.int64)
    
    train_val_feats5 = np.concatenate([feats5[:test_offset_start5, :], feats5[test_offset_end5:, :]], axis=0)
    train_val_texts_id5 = np.concatenate([texts_id5[:test_text_offset_start5, :], texts_id5[test_text_offset_end5:, :]], axis=0)
    train_val_texts_mask5 = np.concatenate([texts_mask5[:test_text_offset_start5, :], texts_mask5[test_text_offset_end5:, :]], axis=0)
    
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
            feats2=train_val_feats2, 
            sizes2=train_val_sizes2, 
            offsets2=train_val_offsets2,
            texts_id2= train_val_texts_id2,
            texts_mask2= train_val_texts_mask2,
            textsizes2= train_val_text_sizes2,
            textoffsets2 = train_val_text_offsets2,
            feats3=train_val_feats3, 
            sizes3=train_val_sizes3, 
            offsets3=train_val_offsets3,
            texts_id3= train_val_texts_id3,
            texts_mask3= train_val_texts_mask3,
            textsizes3= train_val_text_sizes3,
            textoffsets3 = train_val_text_offsets3,
            feats4=train_val_feats4, 
            sizes4=train_val_sizes4, 
            offsets4=train_val_offsets4,
            texts_id4= train_val_texts_id4,
            texts_mask4= train_val_texts_mask4,
            textsizes4= train_val_text_sizes4,
            textoffsets4 = train_val_text_offsets4,
            feats5=train_val_feats5, 
            sizes5=train_val_sizes5, 
            offsets5=train_val_offsets5,
            texts_id5= train_val_texts_id5,
            texts_mask5= train_val_texts_mask5,
            textsizes5= train_val_text_sizes5,
            textoffsets5 = train_val_text_offsets5,
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
            feats2=train_val_feats2, 
            sizes2=train_val_sizes2, 
            offsets2=train_val_offsets2,
            texts_id2= train_val_texts_id2,
            texts_mask2= train_val_texts_mask2,
            textsizes2= train_val_text_sizes2,
            textoffsets2 = train_val_text_offsets2,
            feats3=train_val_feats3, 
            sizes3=train_val_sizes3, 
            offsets3=train_val_offsets3,
            texts_id3= train_val_texts_id3,
            texts_mask3= train_val_texts_mask3,
            textsizes3= train_val_text_sizes3,
            textoffsets3 = train_val_text_offsets3,
            feats4=train_val_feats4, 
            sizes4=train_val_sizes4, 
            offsets4=train_val_offsets4,
            texts_id4= train_val_texts_id4,
            texts_mask4= train_val_texts_mask4,
            textsizes4= train_val_text_sizes4,
            textoffsets4 = train_val_text_offsets4,
            feats5=train_val_feats5, 
            sizes5=train_val_sizes5, 
            offsets5=train_val_offsets5,
            texts_id5= train_val_texts_id5,
            texts_mask5= train_val_texts_mask5,
            textsizes5= train_val_text_sizes5,
            textoffsets5 = train_val_text_offsets5
        )

        train_dataset, val_dataset = random_split(train_val_dataset, [train_nums, val_nums])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)
    
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, 
                            num_workers=4, pin_memory=True, shuffle=False)
    
        return train_loader, val_loader, test_loader