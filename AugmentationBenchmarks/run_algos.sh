# 30 eps
# no aug
python cifar.py --dataset cifar100 --arch wrn --depth 28 --widen-factor 10 --epochs 150
cp ./checkpoint/log.txt ./results/cifar100_noaug_wrn_150ep.txt 

# RE
python cifar.py --dataset cifar100 --arch wrn --depth 28 --widen-factor 10 --epochs 150 --p 0.5
cp ./checkpoint/log.txt ./results/cifar100_RE_wrn_150ep.txt 

# mixup
python cifar.py --dataset cifar100 --arch wrn --depth 28 --widen-factor 10 --epochs 150 --mixup True  --alpha 0.2
cp ./checkpoint/log.txt ./results/cifar100_mixup_wrn_150ep.txt 

#cutmix
python cifar.py --dataset cifar100 --arch wrn --depth 28 --widen-factor 10  --epochs 150 --cutmix True  --beta 1 --cutmix_prob 0.5
cp ./checkpoint/log.txt ./results/cifar100_cutmix_wrn_150ep.txt 

#RandAug
python cifar.py --dataset cifar100 --arch wrn --depth 28 --widen-factor 10  --epochs 150 --rand_augment True
cp ./checkpoint/log.txt ./results/cifar100_randaug_wrn_150ep.txt 

