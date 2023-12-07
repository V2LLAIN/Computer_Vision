import torch
from config import args
from Dataset import CityscapesDataset
from torch.utils.data import DataLoader

ROOT_DIR = args.ROOT_DIR
RESUME = args.RESUME
BATCH_SIZE = args.BATCH_SIZE
TEST_BATCH_SIZE = args.TEST_BATCH_SIZE

train_dataset = CityscapesDataset(split='train',
                                      root_dir=ROOT_DIR,
                                      mode='fine',
                                      eval=False)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_dataset = CityscapesDataset(split='val',
                                      root_dir=ROOT_DIR,
                                      mode='fine',
                                      eval=True)

test_loader = DataLoader(test_dataset,
                         batch_size=TEST_BATCH_SIZE,
                         shuffle=False)


# 데이터 구성 살펴보기 1
print("train data 개수 : {}".format(len(train_dataset)))
print("test data 개수 : {}".format(len(test_dataset)))
# Train 기준

cnt = 0
for x, y in train_dataset:
    print('입력 영상 구조: {}'.format(x.shape)) # 3 x H x W RGB 영상
    print('정답 영상 구조: {}'.format(y.shape)) # H X W label map
    cnt += 1
    if(cnt == 1):
        break