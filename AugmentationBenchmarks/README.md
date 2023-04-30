# RA-DL-Phase1

## To enable mixup data augmentation, use 2 flags, --mixup True --alpha 0.2. Alpha is mixup interpolation coefficient (default: 1).

### Examples:

#### CIFAR10

ResNet-20 baseline on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20
    ```
    
ResNet-20 + Random Erasing on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20 --p 0.5
    ```

wideResNet baseline on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch wrn --depth 28 --widen-factor 10 
    ```

wideResNet + Random Erasing on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch wrn --depth 28 --widen-factor --p 0.5
    ```

#### CIFAR100

ResNet-20 baseline on CIFAR100：
    ```
    python cifar.py --dataset cifar100 --arch resnet --depth 20
    ```
    
ResNet-20 + Random Erasing on CIFAR100：
    ```
    python cifar.py --dataset cifar100 --arch resnet --depth 20 --p 0.5
    ```

wideResNet baseline on CIFAR10：
    ```
    python cifar.py --dataset cifar100 --arch wrn --depth 28 --widen-factor 10 
    ```

wideResNet + Random Erasing on CIFAR10：
    ```
    python cifar.py --dataset cifar100 --arch wrn --depth 28 --widen-factor --p 0.5


