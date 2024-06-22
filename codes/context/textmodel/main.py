import os
import logging
from datetime import datetime
import torch
from torch import nn, optim
from dataset import load_ssl_features, train_valid_test_iemocap_dataloader
from model import BertClassifier, validate_and_test
import numpy as np

logger = logging.getLogger(__name__)
label_dict={'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
sentences = []
dataset = load_ssl_features(
    '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/context/datastore/predict/Session_predict',
    '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/context/datastore/predict/bert',
    '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/context/datastore/predict/bertmask',
    label_dict,
    )
fold_sizes = [1085, 1023, 1151, 1031, 1241] # Session1, 2, 3, 4, 5
fold_list = [0, 1, 2, 3, 4]
test_wa_avg, test_ua_avg, test_f1_avg = 0., 0., 0.
log_name = 'text_emotion_recognition_0'
basepath = "/mnt/cloudstorfs/sjtu_home/zhengshun.xia/need/codes/context/textmodel"
os.makedirs('outputs', exist_ok=True)
log_path = os.path.join(basepath, 'outputs/', log_name) 
logging.basicConfig(
      format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
      level=os.environ.get("LOGLEVEL", "INFO").upper(),
      filename=log_path,
  )
save_dir = '/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/context/textmodel/checkpoint'
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
            eval_is_test= False,
        )
    device = torch.device("cuda:1")
    torch.cuda.empty_cache()
    model = BertClassifier()
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
            ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
            text = net_input['texts_id']
            mask = net_input['texts_mask']

            text = text.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(text, mask)
        
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