import numpy as np
from config import args
import matplotlib.pyplot as plt
from Dataloader import train_dataset

class_table = args.class_table
color_table = args.color_table

def plot_results(output_list=[], output_name_list=[], cols=3):
    plt.rcParams['figure.figsize'] = (18, 8)
    rows = 1

    for i in range(cols):
        image_index = i + 1
        ttile = '{}'.format(output_name_list[i])
        plt.subplot(rows, cols, image_index)
        plt.title(ttile)

        if output_list[i].ndim == 3:
            plt.imshow(output_list[i])
        else:
            plt.imshow(output_list[i], cmap='gray')

    plt.show()
    return

def make_seg_map(pred):
    class_list = np.unique(pred)
    h, w = pred.shape

    seg_map = np.zeros((h, w, 3), dtype=np.uint8)

    class_num = len(class_list)
    for i in range(class_num):
        class_label = class_list[i]
        if class_label == 255:
            continue

        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask = (pred == class_label)

        color_map = color_table[class_label]
        class_name = class_table[class_label]

        r = color_mask[:, :, 0]
        g = color_mask[:, :, 1]
        b = color_mask[:, :, 2]

        r[mask] = color_map[0]
        g[mask] = color_map[1]
        b[mask] = color_map[2]

        class_map = np.dstack((r, g, b))
        seg_map += class_map
    return seg_map


# 데이터 구성 살펴보기 2
# 이미지의 정답을 나타내기 위한 변수
# Train 기준
for x, y in train_dataset:
    input_x = np.array(x.permute(1, 2, 0))
    label_map =  np.array(y)
    semantic_map = make_seg_map(label_map)
    plot_results(output_list=[input_x, label_map, semantic_map]
               , output_name_list=['input', 'label_map', 'semantic_map'], cols=3)
    break