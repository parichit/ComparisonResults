python ../labeling.py --data_set modcloth --debug False --seed 0
python ../train_model.py --data_set modcloth --debug False --seed 0 --labeling_algorithm snorkel
python ../train_model.py --data_set modcloth --debug False --seed 0 --labeling_algorithm triplet