import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import get_act_layer

from triplet_attention import TripletAttention
sigmoid = nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
    
class GeMP(nn.Module):
    def __init__(self, p=3., eps=1e-6, learn_p=False):
        super().__init__()
        self._p = p
        self._learn_p = learn_p
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.set_learn_p(flag=learn_p)

    def set_learn_p(self, flag):
        self._learn_p = flag
        self.p.requires_grad = flag

    def forward(self, x):
        x = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

        return x
    
class LSTM(nn.Module):
    def __init__(self, LSTM_UNITS = 512 , bidirectional=True , batch_first = True,dropout=0.35,out_dim = 2):
        super().__init__()
        self.bn = m = nn.BatchNorm1d(16)
        self.lstm1 = nn.LSTM(64, LSTM_UNITS, bidirectional=bidirectional, batch_first=batch_first,dropout=dropout)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=bidirectional, batch_first=batch_first,dropout=dropout)
        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.head_lstm = nn.Linear(LSTM_UNITS*2, 2)
        self.s1d = Swish_Module()
        self.s2d = Swish_Module()
        self.avd1d = nn.AdaptiveAvgPool1d(1)
        self.head_lstm = nn.Linear(LSTM_UNITS*2, out_dim)
        self.drop_rate = dropout
    def forward(self, x_lstm):
        if len(x_lstm.shape) == 4:
            embedding = x_lstm[:,0,:,:]
        else:
            embedding = x_lstm
        embedding = self.bn(embedding)
        self.lstm1.flatten_parameters()
        h_lstm1, _ = self.lstm1(embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(h_lstm1)
        h_conc_linear1  = self.linear1(h_lstm1)
        h_conc_linear2  = self.linear2(h_lstm2)
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2
        if self.drop_rate:
            hidden = F.dropout(hidden, p=float(self.drop_rate)*2, training=self.training)
        hidden2 = self.avd1d(hidden.transpose(-1,-2))
        hidden_lstm = hidden2.view(embedding.size(0), -1)
        y = self.head_lstm(hidden_lstm)
        return hidden_lstm,y
    
class TABULAR(nn.Module):
    def __init__(self, in_dim=3,out_dim = 2,n_meta_dim=[512, 128]):
        super().__init__()
        self.meta = nn.Sequential(
                nn.Linear(in_dim, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.5),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
        self.head = nn.Linear(n_meta_dim[1], out_dim)

    def forward(self, meta):
        x = self.meta(meta)
        y = self.head(x)
        return x,y

class BlockAttentionModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        CFG,
        n_features: int,
        LSTM_UNITS:int = 512,
    ):
        """Initialize"""
        super(BlockAttentionModel, self).__init__()
        self.backbone = backbone
        self.n_features = n_features
        self.drop_rate = 0.3
        self.pooling = 'gem'
        act_layer = nn.ReLU
        self.CFG = CFG

        self.attention = TripletAttention(self.n_features,
                                              act_layer=act_layer,
                                              kernel_size=13)

        if self.pooling == 'avg':
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        elif self.pooling == 'gem':
            self.global_pool = GeMP(p=4.0, learn_p=False)
        elif self.pooling == 'max':
            self.global_pool = torch.nn.AdaptiveMaxPool2d(1)
        elif self.pooling == 'nop':
            self.global_pool = torch.nn.Identity()
        else:
            raise NotImplementedError(f'Invalid pooling type: {self.pooling}')
        self.LSTM = LSTM(LSTM_UNITS = CFG.LSTM.LSTM_UNITS , bidirectional=CFG.LSTM.bidirectional , batch_first = CFG.LSTM.batch_first,dropout=CFG.LSTM.dropout,out_dim = CFG.num_classes)
        self.TABULAR = TABULAR(in_dim=CFG.TABULAR.in_dim,out_dim = CFG.num_classes,n_meta_dim=CFG.TABULAR.n_meta_dim)
        self.head = nn.Linear(self.n_features+CFG.LSTM.LSTM_UNITS*2+CFG.TABULAR.n_meta_dim[1], CFG.num_classes)
        self.head_cnn = nn.Linear(self.n_features, 2)
        self.train_lstm = CFG.train_lstm
        self.train_cnn = CFG.train_cnn
        self.train_tabular = CFG.train_tabular
        
        
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        if type(self.fc.bias) == torch.nn.parameter.Parameter:
            nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


    def forward(self, x ,x_lstm=None, x_meta=None, gradcam=False):
        """Forward"""
        #lstm
        f = []
        results = []
        if self.train_lstm:
            hidden_lstm,y_lstm = self.LSTM(x_lstm)
            f.append(hidden_lstm)
            results.append(y_lstm)
        #cnn
        if self.train_cnn:
            x = self.backbone(x)
            x = self.attention(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            if gradcam:
                return x
            y_cnn = self.head_cnn(x)
            f.append(x)
            results.append(y_cnn)
        if self.train_tabular:
            x_meta,y_meta = self.TABULAR(x_meta)
            f.append(x_meta)
            results.append(y_meta)
        x = torch.cat(f, dim=1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        if self.train_tabular and self.train_cnn and self.train_tabular:
            y = self.head(x)
        else:
            for i,row in enumerate(results):
                if i==0:
                    y = row
                else:
                    y = y + row
            y = y/len(results)
        return y


def get_model(CFG,backbone_name):
    # create backbone

    if backbone_name in ['tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns',
                         'tf_efficientnet_b2_ns', 'tf_efficientnet_b3_ns',
                         'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns',
                         'tf_efficientnet_b6_ns', 'tf_efficientnet_b7_ns']:
        kwargs = {}
        if True:
            act_layer = get_act_layer('swish')
            kwargs['act_layer'] = act_layer

        backbone = timm.create_model(backbone_name, pretrained=True,in_chans=3, **kwargs)
        n_features = backbone.num_features
        backbone.reset_classifier(0, '')

    else:
        raise NotImplementedError(f'not implemented yet: {backbone_name}')

    model = BlockAttentionModel(backbone,CFG, n_features)

    return model
