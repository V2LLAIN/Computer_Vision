# gradient histogram




# CIFAR-10 Result




# CIFAR-100 Result






# LeNet-Model
- Effectiveness of Scaled Exponentially-Regularized Linear Units (SERLUs)을 기반으로 실험을 진행
- (https://arxiv.org/abs/1807.10117v2)
- 위의 논문을 기반으로 실험하여 수정된 LeNet모델을 사용.

- 6 Layers with 4 Conv layers and 2 FC layers
- Optimization Algorithm: RMSProp with Learning_Rate=1e-4, decay=1e-6
- 200 epochs with 128 batch sizes

- Data Augmentation: flipping images vertically and horizontally
