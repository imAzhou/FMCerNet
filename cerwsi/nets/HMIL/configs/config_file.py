# dataset
dataset_name = 'LCerScan'
data_root = "/path/to/dataset"
splits = "/path/to/splits"
folds = "0"
label_list = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']
mapping = "0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2" # mapping from fine-grained labels to coarse-grained labels

# classification
n_class = [3, 7]

# resume training
pretrained_path = ""
log_folder = "log"

# optimization, training, and testing
lr = 1e-4
num_epochs = 200
train_batch_size = 64
test_batch_size = 1
num_workers = 4
seed = 666

# GPU
gpus = "2" # GPU ID to use