import os
import logging
from datetime import datetime
import torch
from torch import nn, optim
from dataset_3 import load_ssl_features, train_valid_test_iemocap_dataloader
from model_3 import BaseModel, validate_and_test

logger = logging.getLogger(__name__)

label_dict={'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
dataset = load_ssl_features(
    '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/context/datastore/predict/Session_predict',
    '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/context/datastore/prev1/Session_prev1',
    '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/context/datastore/prev2/Session_prev2',
    '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/context/datastore/prev3/Session_prev3',
    label_dict
    )
fold_sizes = [1085, 1023, 1151, 1031, 1241] # Session1, 2, 3, 4, 5
fold_list = [0, 1, 2, 3, 4]
test_wa_avg, test_ua_avg, test_f1_avg = 0., 0., 0.
log_name = 'speech_emotion_recognition_3'
basepath = "/mnt/cloudstorfs/sjtu_home/zhengshun.xia/need/codes/context/speechmodel"
os.makedirs('outputs', exist_ok=True)
log_path = os.path.join(basepath, 'outputs/', log_name) 
logging.basicConfig(
      format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
      level=os.environ.get("LOGLEVEL", "INFO").upper(),
      filename=log_path,
  )
save_dir = '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/context/speechmodel/checkpoint'
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(os.path.join(save_dir, timestamp_str), exist_ok=True)
save_dir = os.path.join(save_dir, timestamp_str, "checkpoint.pt")
for fold in fold_list:
    test_len = fold_sizes[fold] 
    test_idx_start = sum(fold_sizes[:fold])
    test_idx_end = test_idx_start + test_len 
    train_loader, val_loader, test_loader = train_valid_test_iemocap_dataloader(
            dataset,
            16,
            test_idx_start,
            test_idx_end,
            eval_is_test=False
        )
    device = torch.device("cuda:2")
    torch.cuda.empty_cache()
    model = BaseModel(
        input_dim=768, 
        output_dim=len(label_dict),
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_wa = 0
    best_val_wa_epoch = 0
    best_val_acc = 0
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"------Now the new task is begin------")
    for epoch in range(100):  # Adjust the number of epochs as per your requirement
        model.train()
        train_loss = 0
        for batch in train_loader:
            ids, net_input1, net_input2, net_input3, net_input , labels = batch["id"], batch["net_input1"], batch["net_input2"], batch["net_input3"],  batch["net_input"], batch["labels"]
            feats = net_input["feats"]
            speech_padding_mask = net_input["padding_mask"]
            feats1 = net_input1["feats1"]
            speech_padding_mask1 = net_input1["padding_mask1"]
            feats2 = net_input2["feats2"]
            speech_padding_mask2 = net_input2["padding_mask2"]
            feats3 = net_input3["feats3"]
            speech_padding_mask3= net_input3["padding_mask3"]
        
            feats = feats.to(device)
            speech_padding_mask = speech_padding_mask.to(device)
            feats1 = feats1.to(device)
            speech_padding_mask1 = speech_padding_mask1.to(device)
            feats2 = feats2.to(device)
            speech_padding_mask2 = speech_padding_mask2.to(device)
            feats3 = feats3.to(device)
            speech_padding_mask3 = speech_padding_mask3.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(feats1, speech_padding_mask1, feats2, speech_padding_mask2 , feats3, speech_padding_mask3, feats, speech_padding_mask)
        
            loss = criterion(outputs, labels.long())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()         
            
        val_wa, val_ua, val_f1 = validate_and_test(model, val_loader, device, num_classes=len(label_dict))   
        if val_wa > best_val_wa:
            best_val_wa = val_wa
            best_val_wa_epoch = epoch
            torch.save(model, save_dir)
        logger.info(f"Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader):.6f}, Validation WA: {val_wa:.2f}%; UA: {val_ua:.2f}%; F1: {val_f1:.2f}%")
    #scheduler.step()
    model = torch.load(save_dir).to(device)
    test_wa, test_ua, test_f1 = validate_and_test(model, test_loader, device, num_classes= 4)
    test_wa_avg += test_wa
    test_ua_avg += test_ua
    test_f1_avg += test_f1
    
    logger.info(f"\n\nThe {fold+1}th Fold at epoch {best_val_wa_epoch + 1}, test WA {test_wa}%; UA {test_ua}%; F1 {test_f1}%")
    
logger.info(f"Average WA: {test_wa_avg/len(fold_list)}%; UA: {test_ua_avg/len(fold_list)}%; F1: {test_f1_avg/len(fold_list)}%")