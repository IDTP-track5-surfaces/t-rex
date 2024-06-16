# TREX and FSRN-CNN codebase. 

Implementation code for the FSRN-CNN by [Thapa et. al](https://ivlab.cse.lsu.edu/pub/fluid_cvpr20.pdf) as well as our interdisciplinary team project's "TREX". 


Find the FSRN-CNN and TREX code in the respective directories. 

For ease of dataloading during training, name the following depth, normal, refracted, and reference files accordingly. 
```
data_dir/data_name/train/depth/depth_id.npy
data_dir/data_name/train/normal/normal_id.npy
data_dir/data_name/train/refraction/reference_name/id.png
data_dir/data_name/train/reference/reference_name.png
```
switch "train" with "validation" to set the validation data set. 

This is how the root data directory should look for instance: 

<img width="162" alt="Screenshot 2024-06-16 at 23 55 38" src="https://github.com/IDTP-track5-surfaces/t-rex/assets/58450012/eb3ec614-bfcb-438f-9036-907bbf609598">

To run a training loop on your data called pool_homemade. Run the following: 
```
python3 run.py --val_data pool_homemade --train_data pool_homemade --root_dir path/to/your/root/data/dir/ 
```
To run inference with a saved model on random samples from a validation set. 
```
python3 run.py --val_data pool_homemade --infer True --load_model /path/to/loaded/model.h5
```

If you would like to train on a dataset's training data and validate on anothers validation data, you could do the following for example:

```
python3 run.py --val_data pool_homemade --train_data dynamic --root_dir path/to/your/root/data/dir/ 
```
Of course, all data should be within the same root data directory. 







