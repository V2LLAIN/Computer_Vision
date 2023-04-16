# CIFAR-10 Result
<img width="1862" alt="스크린샷 2023-04-15 오후 10 15 08" src="https://user-images.githubusercontent.com/104286511/232226137-acf65e05-85b6-4026-a3f4-886b35fb7bc6.png">


# CIFAR-100 Result



# ResNet-model
ResNet모델은 총 3개의 스택(stacks)으로 이뤄져 있으며, 각 스택은 여러 개의 residual block으로 구성되어 있습니다.
- 이 코드에서는 각 residual block이 2개의 convolution layer로 이뤄져 있고, 
- 스택마다 residual block의 수를 조절할 수 있는 인자가 있기 때문에, 
- 총 몇 층으로 이뤄져 있는지는 size, stacks, starting_filter 등의 인자에 따라 달라진다. 

이 코드에서는 size=44, stacks=3으로 설정되어 있으므로, 총 약 44개의 convolution layer로 이뤄지며,
residual_blocks는 (size - 2) // 6로 계산되므로, 이 코드에서는 44개의 Residual Block을 가지고 있습니다. 
이는 size가 44이고, 각각의 스택(Stack)에서 6개의 레이어로 이뤄진 Residual Block이 3개 쌓인 구조이기 때문입니다. 
따라서 이 ResNet 모델은 총 44 * 3 + 2 = 134개의 레이어를 갖습니다.
