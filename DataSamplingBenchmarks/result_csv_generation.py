import numpy as np
import pandas as pd
import sys
import os
import json
import pickle
import csv
exps_dir=sys.argv[1]
dir_names=os.listdir(exps_dir)
#dataset,no.of datasamples,fraction of data,scored by,test acc,train acc
runs_dirs=[]
for dir in os.listdir(exps_dir):
    if('run' in dir):
        runs_dirs.append(os.path.join(exps_dir,dir))
    elif('size' in dir):
        for inner_dir in os.listdir(os.path.join(exps_dir,dir)):
            if('run' in inner_dir):
                runs_dirs.append(os.path.join(exps_dir,dir,inner_dir))

for run_dir in runs_dirs:
    args_path=os.path.join(run_dir,'args.json')
    with open(args_path) as f:
        args_json=json.load(f)
    
    recorder_path=os.path.join(run_dir,'recorder.pkl')
    with open(recorder_path,"rb") as f :
        recorder=pickle.load(f)

    exec_time_path=os.path.join(run_dir,'exec_time.txt')
    with open(exec_time_path,"r") as f:
        exec_time=f.read()
    result_row=[]
    result_row.append(args_json['dataset'])
    result_row.append(args_json['model'])
    result_row.append(args_json['scores_path'])
    result_row.append(args_json['num_train_examples'])
    result_row.append(args_json['num_train_examples/50000'])
    result_row.append(max(recorder['train_acc']))
    result_row.append(max(recorder['test_acc']))
    result_row.append(exec_time)


    if (not os.path.exists('./results.csv')):
        with open("./results.csv", 'a') as results_csv:
            csvwriter = csv.writer(results_csv)
            csvwriter.writerow(['Dataset',	'Neural network', 'Scored by',
                               'No.of train datasamples','fraction of data trained on',	'Train acc',	'Test acc', 'Execution time (in secs)'])
    with open("./results.csv", 'a') as results_csv:
        csvwriter = csv.writer(results_csv)
        csvwriter.writerow(result_row)
    




    


