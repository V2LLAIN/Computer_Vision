# gradient histogram
<img width="476" alt="스크린샷 2023-04-28 오후 6 59 48" src="https://user-images.githubusercontent.com/104286511/235118064-0d362dec-4af5-409b-aa2d-46c4f5605f8a.png">

![히스토그램](https://user-images.githubusercontent.com/104286511/235304240-9baff9e3-1f1d-464d-bca2-76c8381f1bc7.png)




# CIFAR-10 Result
<img width="1237" alt="스크린샷 2023-04-22 오전 9 47 33" src="https://user-images.githubusercontent.com/104286511/233755492-51e2a9ff-44fc-4e87-9457-7d78a2e2f424.png">
<img width="880" alt="스크린샷 2023-04-22 오전 10 49 18" src="https://user-images.githubusercontent.com/104286511/233755579-7b1ed4ce-4cd5-4bff-96c1-ffb4aa25ad9a.png">


# CIFAR-100 Result
<img width="1235" alt="CIFAR100" src="https://user-images.githubusercontent.com/104286511/233762793-fbb284af-6895-4181-b3b9-7ec96c005a87.png">
<img width="926" alt="스크린샷 2023-04-22 오후 1 38 59" src="https://user-images.githubusercontent.com/104286511/233762859-f1faa61a-05cd-4825-a293-10eb1de23431.png">


# Activation graph
![graph](https://user-images.githubusercontent.com/104286511/235304305-bec3e9bc-e3d4-49f6-bd07-59c60da64ca2.png)


# LeNet-Model
- Effectiveness of Scaled Exponentially-Regularized Linear Units (SERLUs)을 기반으로 실험을 진행
- (https://arxiv.org/abs/1807.10117v2)
- 위의 논문을 기반으로 실험하여 수정된 LeNet모델을 사용.

- 6 Layers with 4 Conv layers and 2 FC layers
- Optimization Algorithm: RMSProp with Learning_Rate=1e-4, decay=1e-6
- 200 epochs with 128 batch sizes

- Data Augmentation: flipping images vertically and horizontally




