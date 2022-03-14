# Data Pre-processing

### Water Treatment Plant Dataset
Follow the jupyter notebook to pre-process water treatment plant dataset - WADI. Follow the notebook to replicate a similar process for the SWaT dataset.

### Server Machine Dataset

For example, use the following command to pre-process machine-1-1
```
# generate data
python generate_training_data.py --output_dir '../data/machine-1-1' --train_path 'machine-1-1_train.pkl' --test_path 'machine-1-1_test.pkl' --anomaly_path 'machine-1-1_test_label.pkl' --window_size 100 --val_ratio 0.3 
```

