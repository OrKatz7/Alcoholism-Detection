import help_utils
from torch.utils.data import DataLoader, Dataset
import albumentations
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import random
import mne

class ICA:
    def __init__(self):
        self.channel_list = ['AF1', 'AF2', 'AF7', 'AF8', 'AFZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ', 'CZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCZ', 'FP1', 'FP2', 'FPZ', 'FT7', 'FT8', 'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO1', 'PO2', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'T8', 'TP7', 'TP8', 'X', 'Y', 'nd']
        self.info = mne.create_info(self.channel_list, 256, ch_types='eeg', verbose=False)
        self.ica = mne.preprocessing.ICA(n_components=64, verbose=False)
        
    def __call__(self,signal):
        raw = mne.io.RawArray(data=signal, info=self.info, verbose=False)
        raw.filter(l_freq=1., h_freq=None, verbose=False);
        self.ica.fit(raw, picks=self.channel_list)
        return self.ica.get_sources(raw).get_data()
    
def get_ica(signal, mode='sources'):
    """
    mode: 
        sources - projected components x signal (64x265)
        components - ICA components (64x64)
    """;
    
    # define channels
    channel_list = ['AF1', 'AF2', 'AF7', 'AF8', 'AFZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ', 'CZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCZ', 'FP1', 'FP2', 'FPZ', 'FT7', 'FT8', 'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO1', 'PO2', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'T8', 'TP7', 'TP8', 'X', 'Y', 'nd']
    
    # create mne raw object
    info = mne.create_info(channel_list, 256, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(data=signal, info=info, verbose=False)

    # filter signal
    raw.filter(l_freq=1., h_freq=None, verbose=False);

    # run ica
    ica = mne.preprocessing.ICA(n_components=64, verbose=False)
    ica.fit(raw, picks=channel_list);
    
    if mode == 'sources':
        return ica.get_sources(raw).get_data()
    elif mode == 'components':
        return norm_min_max(ica.get_components())
    else:
        raise NotImplementedError
        

    
class CutoutV2(albumentations.DualTransform):
    def __init__(
        self,
        num_holes=8,
        max_h_size=8,
        max_w_size=8,
        fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        super(CutoutV2, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def apply(self, image, fill_value=0, holes=(), **params):
        return albumentations.functional.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")
    
def get_transforms(*, data,CFG):
    
    if data == 'train':
        return albumentations.Compose([
            albumentations.PadIfNeeded(min_height=CFG.CNN.pad_size, min_width=CFG.CNN.pad_size),
            albumentations.RandomBrightness(p=0.75),
            albumentations.RandomContrast(p=0.75),
#             albumentations.HorizontalFlip(p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.1),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, border_mode=0, p=0.5),
            CutoutV2(max_h_size=int(CFG.CNN.size * 0.2), max_w_size=int(CFG.CNN.size * 0.2), num_holes=1, p=0.8),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return albumentations.Compose([
            albumentations.PadIfNeeded(min_height=CFG.CNN.size, min_width=CFG.CNN.size),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
def read_img(file_path,size,fn,type_img='img'):
    img = np.load(file_path)['img']
    if type_img == 'ica':
        img[:,:,0] = np.load(file_path)['ica'][:,:,0]
    img-=img.min()
    img/=img.max()
    image = np.zeros([size,size,3])
    for row in range(size):
        image[:,row] = img[:,row//4]
    image*=255
    image = image.astype(np.uint8).transpose(1,0,2)
    return image      

class TrainDataset(Dataset):
    def __init__(self, df, lstm_df=None, tabular_df = None ,config=None, transform=None):
        self.df = df
        self.lstm_df = lstm_df
        self.tabular_df = tabular_df
        self.config = config
        self.idxs = df['id'].values
        self.trail = df['trial'].values
        self.file_names = df['path'].values
        self.labels = df['class'].values
        self.stimuli =  df['stimuli'].values
        self.map ={}
        for i,row in enumerate(config.DATA.stimuli):
            self.map[row] = i
        self.transform = transform
        self.tensor = ToTensorV2()
        self.size = self.config.CNN.size
        self.columns = self.config.LSTM.columns
        self.img_pp_fun= None
        self.tabular_pp_fun= None
        self.lstm_pp_fun= None
        if self.config.CNN.pp_fun is not None:
            print(f"config.CNN.pp_fun {config.CNN.pp_fun}")
            self.img_pp_fun = eval(self.config.CNN.pp_fun)
        if self.config.TABULAR.pp_fun is not None:
            self.tabular_pp_fun = eval(self.config.TABULAR.pp_fun)
        if self.config.LSTM.pp_fun is not None:
            self.lstm_pp_fun = eval(self.config.LSTM.pp_fun)
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = file_name.replace("/images/","/images_ica/")
        label = torch.tensor(self.labels[idx]).long()
        if self.config.train_tabular:
            meta = self.map[self.stimuli[idx]]
            if self.tabular_pp_fun:
                meta = self.tabular_pp_fun(meta)
            meta = torch.tensor(meta).float()
        else:
            meta = torch.empty([0])
        if self.config.train_cnn:
            image = read_img(file_path,self.size,self.img_pp_fun,self.config.CNN.type_image)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
        else:
            image = torch.empty([0])
        if self.config.train_lstm:
            if self.config.LSTM.use_sax:
                t = self.trail[idx]
                n = self.idxs[idx]
                x_lstm = self.lstm_df[(self.lstm_df['id'] == n) & (self.lstm_df['trial'] == t)]
                x_lstm = torch.tensor(x_lstm[self.columns].values.astype(float)).float()
            else:
                x_lstm = np.load(file_path)['img'][:,:,0]
            if self.lstm_pp_fun:
                x_lstm = self.lstm_pp_fun(x_lstm)
            
        else:
            x_lstm = torch.empty([0])
        return image,x_lstm,meta, label,file_name
