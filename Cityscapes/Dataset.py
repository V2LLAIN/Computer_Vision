import torch
import numpy as np
import os, cv2, natsort
from torch.utils.data import Dataset

class CityscapesDataset(Dataset):
    def __init__(self, split, root_dir, target_type='semantic', mode='fine', transform=None, eval=False):
        self.transform = transform

        if mode == 'fine':
            self.mode = 'gtFine'

        elif mode == 'coarse':
            self.mode = 'gtCoarse'

        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = eval

        self.label_path = os.path.join(root_dir, self.mode, self.split)
        self.rgb_path = os.path.join(root_dir,'leftImg8bit', self.split)

        # 알파벳 순서대로 해당 도시들을 정렬
        city_list = natsort.natsorted(os.listdir(self.label_path))

        for city in city_list:
            city_path = os.path.join(self.label_path, city)

            temp = natsort.natsorted(os.listdir(city_path))
            list_items = temp.copy()

            """
            # cityscapes는 원래 30개의 class가 있으나 일반적으로 사용되는 class는 19개
            # 아래 코드는 일반적으로 많이 사용되는 19개의 class를 선별하는 코드
            """
            for item in temp:
                if not item.endswith('labelTrainIds.png', 0, len(item)):
                    list_items.remove(item)

            # 선별된 19개 class label을 가진 맵을 리스트에 저장
            list_items = [os.path.join(city, path) for path in list_items]

            self.yLabel_list.extend(list_items)

            self.XImg_list.extend(
                [os.path.join(city, path) for path in natsort.natsorted(os.listdir(os.path.join(self.rgb_path, city)))]
            )

            # 입력 데이터의 개수와 정답 데이터의 개수가 일치하는지 확인
            assert len(self.yLabel_list) == len(self.XImg_list)


    def __len__(self):
        # 전체 데이터 개수 반환
        length = len(self.XImg_list)
        return length

    def __getitem__(self, index):
        input_path = os.path.join(self.rgb_path, self.XImg_list[index])
        input_name = input_path.split('/')[-1][:-16]
        gt_path = os.path.join(self.label_path, self.yLabel_list[index])
        gt_name = gt_path.split('/')[-1][:-25]

        assert input_name == gt_name

        image = cv2.imread(input_path)[:, :, ::-1]
        image = image.astype(np.float32)
        y = cv2.imread(gt_path)[:, :, 0]

        """
        # cityscapes는 1024 x 2048의 고해상도 이미지
        # 원활한 딥러닝 학습을 하기 위해 4배 줄여서 학습
        # 입력 및 정답 영상 모두 256 x 512
        """
        h, w, _ = image.shape
        image = cv2.resize(image, dsize=(512, 256), interpolation=cv2.INTER_NEAREST) # resize 시 ( w, h ) 순으로 입력
        y = cv2.resize(y, dsize=(512, 256), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            image = self.transform(image)
            y = self.transform(y)

        # 0 ~ 1 Normalization
        image = image / 255

        # numpy -> tensor
        image = torch.from_numpy(image)
        y = torch.from_numpy(y)

        # H x W x C -> C x H x W
        image = image.permute(2, 0, 1)

        # 정수형 타입으로 변경
        y = y.type(torch.LongTensor)

        # evaluation
        if self.eval:
            return image, y, input_name

        # trainset
        else:
            return image, y