import cv2, torch, os
from tqdm import tqdm
from torch.nn import functional as F
from train import UNet_with_Attention
from Dataloader import test_loader
from segmentation_map import *

new_model = UNet_with_Attention().to(args.DEVICE)
checkpoint_path = args.checkpoint_path
checkpoint = torch.load(checkpoint_path)
new_model.load_state_dict(checkpoint)
print("사전 학습된 가중치 업로드 완료 ")

new_model.eval()

save_path = args.test_save_path
os.makedirs(save_path, exist_ok=True)

with torch.no_grad():
  for idx, batch in enumerate(tqdm(test_loader)):
    X, y, name = batch # here 's' is the name of the file stored in the root directory
    X, y = X.to(args.DEVICE), np.array(y.to('cpu')[0])
    predictions = new_model(X)

    predictions = F.softmax(predictions, dim=1)
    pred_labels = torch.argmax(predictions, dim=1)
    pred_labels = pred_labels.float()
    pred_labels = pred_labels.to('cpu')
    pred_labels = np.array(pred_labels[0])
    pred_seg_map = make_seg_map(pred_labels)
    gt_seg_map = make_seg_map(y)
    name = str(name[0])
    cv2.imwrite(os.path.join(save_path, name + '.png'), pred_seg_map[:, :, ::-1])

    # 예측 맵과 정답 맵 비교 - 5 개만 선정
    if idx in [0, 1, 2, 3, 4]:
      plot_results(output_list=[pred_seg_map, gt_seg_map], output_name_list=['pred', 'gt'], cols=2)
