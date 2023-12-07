import argparse
from table import *
parser = argparse.ArgumentParser()

# Dataloader.py
parser.add_argument('--ROOT_DIR', default="./cityscapes/")
parser.add_argument('--RESUME', type=bool, default=False)
parser.add_argument('--BATCH_SIZE', type=int, default=8)
parser.add_argument('--TEST_BATCH_SIZE', type=int, default=1)

# segmentation_map.py
parser.add_argument('--class_table', default=str(class_table))
parser.add_argument('--color_table', default=str(color_table))

# train.py
parser.add_argument('--DEVICE', default="cuda")
parser.add_argument('--save_path', default="./output")
parser.add_argument('--LEARNING_RATE', default="5e-4")
parser.add_argument('--EPOCHS', type=int, default=100)
parser.add_argument('--class_num', type=int, default=19+1)
parser.add_argument('--smooth', type=float, default=1e-10)

# eval.py
parser.add_argument('--checkpoint_path', default='/root/Study/Computer Vision/output/best.pth')
parser.add_argument('--save_path', default='./test_result')

args = parser.parse_args()



"""

parser.add_argument('--', type=int, default=)
"""
