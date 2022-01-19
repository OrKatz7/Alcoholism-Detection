import yaml

class data_config:
    sax_csv_path = "/sise/liorrk-group/OrDanOfir/eeg/data/dataset_change_point.parquet"
    img_csv_path = "/sise/liorrk-group/OrDanOfir/eeg/data/img_train.csv"
    OUTPUT_DIR = './output'
    num_workers=8
    batch_size=16
    pin_memory=True
    drop_last=True
    stimuli = ['S2 match']
    drop_stimuli = ['S2 nomatch err','S2 match err','S1 obj','S2 nomatch']
    cnn_data_pp = None
    lstm_data_pp = None
    tabular_data_pp = None
    train_image_augment = 'train_cnn_aug.yaml'
    val_imag_augment = 'val_cnn_aug.yaml'
    train_val_datasets = 'data.TrainDataset'
    image_mix_up = 0.0
    split_by ='id'


class image_model_config:
    pad_size = 256
    size=256
    triplet_attention = True
    attention_kernel_size=13
    act_layer = 'swish'
    in_chans=3
    pooling = 'gem'
    drop_rate = 0.3
    flatten=True
    pp_fun = None
    type_image = 'ica'

class lstm_config:
    LSTM_UNITS = 512
    bidirectional=True
    batch_first=True
    dropout=0.1
    columns = ['AF1', 'AF2', 'AF7', 'AF8', 'AFZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
       'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ', 'CZ', 'F1', 'F2', 'F3',
       'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
       'FCZ', 'FP1', 'FP2', 'FPZ', 'FT7', 'FT8', 'FZ', 'O1', 'O2', 'OZ', 'P1',
       'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO1', 'PO2', 'PO7', 'PO8',
       'POZ', 'PZ', 'T7', 'T8', 'TP7', 'TP8', 'X', 'Y', 'nd']
    pp_fun = None
    use_sax = True

class tabular_config:
    n_meta_dim=[512, 128]
    dropout=0.25
    pp_fun = 'help_utils.on_hot'
    in_dim = 3

class torch_config:
    scheduler='torch.optim.lr_scheduler.CosineAnnealingWarmRestarts' 
    scheduler_args = dict(T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
    gradient_accumulation_steps=1
    max_grad_norm=1000
    loss_fn = 'torch.nn.CrossEntropyLoss'
    loss_args = dict()
    optimizer = 'torch.optim.Adam'
    optimizer_args = dict(lr=1e-4, weight_decay=1e-6, amsgrad=False)


class Config:
    exp_name = 'ensemble'
    num_classes=2
    k_fold_fun = 'help_utils.split_kfold'
    model_fn = 'models.get_model'
    model_args = dict(backbone_name = 'tf_efficientnet_b0_ns')
    debug=False
    verbose = True
    print_freq=100
    epochs=15
    seed=42
    target_size=1
    target_col='class'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
    inference=False
    score_metric = 'sklearn.metrics.accuracy_score'
    train_lstm = True
    train_cnn = False
    train_tabular = False
    LSTM = lstm_config
    CNN = image_model_config
    TABULAR = tabular_config
    TORCH = torch_config
    DATA = data_config

def load_config(config_file):
    with open(config_file, 'r') as fp:
        opts = yaml.safe_load(fp)
    return Config(**opts)
