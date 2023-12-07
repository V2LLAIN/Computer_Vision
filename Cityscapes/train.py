import os, torch, tqdm, datetime
import numpy as np
import torch.nn as nn
from config import args
from torch.nn import functional as F
from model import UNet_with_Attention
from Dataloader import train_dataset, test_dataset, train_loader, test_loader

DEVICE = args.DEVICE
LEARNING_RATE = args.LEARNING_RATE
model = UNet_with_Attention().to(args.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
print(model)

def train(epoch=0):
    log = open(os.path.join(save_path,'log.txt'),'w')
    log.write('Epoch\tLoss\tmIoU\n')
    log.close()

    print("#####################################################################")
    print("Train information")
    print("batch size: {}".format(args.BATCH_SIZE))
    print("Learning rate: {}".format(LEARNING_RATE))
    print("Total Epochs: {}".format(EPOCHS))
    print("Train dataset number: {}".format(len(train_dataset)))
    print("Test dataset number: {}".format(len(test_dataset)))
    print("#####################################################################")

    best_mean_iou_score = 0.0
    for epoch in range(epoch, EPOCHS):
        print("Current epoch {} start".format(epoch + 1))
        total_loss = 0.0
        iteration = 0.0

        model.train()

        for batch in tqdm(train_loader):

            X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X)

            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            total_loss += loss.item()

            print('[%s] [%d/%d], CrossEntropyLoss per iter : %f'
                % (datetime.datetime.now().strftime("%m%d_%H:%M"), iteration, len(train_loader),
                   loss.item()))

        print('[%s] [%d/%d], CrossEntropyLoss: %f'
               % (datetime.datetime.now().strftime("%m%d_%H:%M"), epoch + 1, EPOCHS,
                   total_loss / iteration))

        total_mean_iou = 0.0

        model.eval()

        for batch in tqdm(test_loader):
            X, y, img_name = batch
            img_name = img_name[0]

            with torch.no_grad():
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)

                pred_mask = F.softmax(pred, dim=1)
                pred_mask = torch.argmax(pred, dim=1) # B x H x W
                pred_mask = pred_mask.contiguous().view(-1) # 1차원 vector로 flatten
                gt_mask = y.contiguous().view(-1) # 1차원 vector로 flatten

                iou_per_class = []

                for class_id in range(class_num): # loop per pixel class
                    true_class = pred_mask == class_id
                    true_label = gt_mask == class_id

                    if true_label.long().sum().item() == 0: # no exist label in this loop
                        iou_per_class.append(np.nan)

                    else:
                        intersect = torch.logical_and(true_class, true_label).sum().float().item()
                        union = torch.logical_or(true_class, true_label).sum().float().item()

                        iou = (intersect + smooth) / (union +smooth)
                        iou_per_class.append(iou)

            total_mean_iou += np.nanmean(iou_per_class)

        pre_mean_iou = (total_mean_iou / len(test_loader))
        pre_mean_iou_score = pre_mean_iou * 100

        print("miou score: {}".format(pre_mean_iou_score))

        # 가중치 저장 update
        if best_mean_iou_score < pre_mean_iou_score:
            best_mean_iou_score = pre_mean_iou_score

            # 모델의 가중치 저장
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))

        log = open(os.path.join(save_path,'log.txt'),'a')
        log.write(f'{epoch+1}\t{total_loss / iteration}\t{pre_mean_iou_score}\n')
        log.close()

if __name__ == '__main__':
    EPOCHS = args.EPOCHS
    class_num = args.class_num
    smooth= args.smooth
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    train(epoch=0)