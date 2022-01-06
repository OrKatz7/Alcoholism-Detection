import help_utils
from torch.utils.data import DataLoader, Dataset
import albumentations
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import random
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
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),
            albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
            CutoutV2(max_h_size=int(CFG.CNN.size * 0.2), max_w_size=int(CFG.CNN.size * 0.2), num_holes=1, p=0.75),
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
    
def read_img(file_path,size):
    img = np.load(file_path)['img']
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
            self.img_pp_fun = eval(self.config.CNN.pp_fun)
        if self.config.TABULAR.pp_fun is not None:
            self.tabular_pp_fun = eval(self.config.TABULAR.pp_fun)
        if self.config.LSTM.pp_fun is not None:
            self.lstm_pp_fun = eval(self.config.LSTM.pp_fun)
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = file_name
        label = torch.tensor(self.labels[idx]).long()
        if self.config.train_tabular:
            meta = self.map[self.stimuli[idx]]
            if self.tabular_pp_fun:
                meta = self.tabular_pp_fun(meta)
            meta = torch.tensor(meta).float()
        else:
            meta = torch.empty([0])
        if self.config.train_cnn:
            image = read_img(file_path,self.size)
            if self.img_pp_fun:
                image = self.img_pp_fun(image)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
        else:
            image = torch.empty([0])
        if self.config.train_lstm:
            t = self.trail[idx]
            n = self.idxs[idx]
            x_lstm = self.lstm_df[(self.lstm_df['id'] == n) & (self.lstm_df['trial'] == t)]
            if self.lstm_pp_fun:
                x_lstm = self.lstm_pp_fun(x_lstm)
            x_lstm = torch.tensor(x_lstm[self.columns].values.astype(float)).float()
        else:
            x_lstm = torch.empty([0])
        return image,x_lstm,meta, label