
# Data Augmentation benchmarks

### cifar.py is the main driver file.

``` --dataset ``` flag defines which dataset to use. options are "cifar10" and "cifar100".

```--arch``` and ``` --depth``` flags are used to select the model to run. To run resnet20 use ```--arch resnet  --depth 20```. For wideresnet, use ```--arch wrn --depth 28```

**Random Erasing**: ``` --p ``` flag is used to enable random erasing Augmentation technique, where p is the random erasing probability. p should be in [0,1] range.

**Mixup**: ``` --mixup``` should be ```True``` and ```--alpha``` is mixup interpolation coefficient (default: 1).

**Cutmix**: ```cutmix``` should be ```True```.  ```--cutmix_prob``` flag is cutmix probability (default:0). ```--beta``` is hyparameter, value range (0,1].

**Random augment**: ```--rand_augment``` should be ```True```. ```--n``` and ```--m``` are n and m value required for Random Augment. defults of n and m are 2 and 10 respectively.

**YOCO**: ```--yoco``` flag should be ```True```

### Examples:

ResNet-20 baseline on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20
    ```
    
ResNet-20 + Random Erasing on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20 --p 0.5
    ```
    
ResNet-20 + Mixup on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20 --mixup True --alpha 0.5
    ```

ResNet-20 + Cutmix on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20 --cutmix True --beta 0.5 --cutmix_prob 0.5
    ```

ResNet-20 + Random augment on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20 --rand_augment True
    ```   

ResNet-20 + YOCO on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20 --yoco True
    ```

WRN baseline on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch wrn --depth 28
    ```
    
WRN + Random Erasing on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch wrn --depth 28 --p 0.5
    ```
    
WRN + Mixup on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch wrn --depth 28 --mixup True --alpha 0.5
    ```

WRN + Cutmix on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch wrn --depth 28 --cutmix True --beta 0.5 --cutmix_prob 0.5
    ```

WRN + Random augment on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch wrn --depth 28 --rand_augment True
    ```   

WRN + YOCO on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch wrn --depth 28 --yoco True
    ```

note: To run cifar100, just change --dataset flag to cifar100.

#### Results will be written to results.csv file for each successful run. 
Dataset, Neural network, Augmentation Algorithm, Epochs, Train acc, Test acc, Execution time (in secs) are the fileds of results.csv
