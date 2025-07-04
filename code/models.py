import torch
import torch.utils.data
import torch.nn as nn

def get_model(params):
    print("model: ", params['model'])
    if params['model'] == 'ResidualFCNet':
        return ResidualFCNet(params['input_dim'], params['num_filts'], params['depth'])
    elif params['model'] == 'ResidualFCNetLatent':
        return ResidualFCNetLatent(params['input_dim'], params['num_filts'], params['depth'])
    elif params['model'] == 'LinNet':
        return LinNet(params['input_dim'])
    else:
        raise NotImplementedError('Invalid model specified.')

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class ResidualFCNet(nn.Module):

    def __init__(self, num_inputs, num_filts, depth=4):
        super(ResidualFCNet, self).__init__()
        self.inc_bias = False
        #self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.val_emb = nn.Linear(num_filts, 1, bias=self.inc_bias)
        layers = []
        layers.append(nn.Linear(num_inputs, num_filts))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers.append(ResLayer(num_filts))
        self.feats = torch.nn.Sequential(*layers)

    def forward(self, x, class_of_interest=None, return_feats=False):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        #if class_of_interest is None:
        #    class_pred = self.class_emb(loc_emb)
        #else:
        #    class_pred = self.eval_single_class(loc_emb, class_of_interest)
        out = self.val_emb(loc_emb)  # shape: (B, 1)
        return out.squeeze(-1) #torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return x @ self.class_emb.weight[class_of_interest, :] + self.class_emb.bias[class_of_interest]
        else:
            return x @ self.class_emb.weight[class_of_interest, :]
        
class ResidualFCNetLatent(nn.Module):

    def __init__(self, num_inputs, num_filts, depth=4):
        super(ResidualFCNetLatent, self).__init__()
        self.use_prev_latent = True
        self.latent_dim = num_filts

        in_dim = num_inputs + (self.latent_dim if self.use_prev_latent else 0)

        self.val_emb = nn.Linear(num_filts, 1, bias=False)
        layers = [nn.Linear(in_dim, num_filts), nn.ReLU(inplace=True)]
        for _ in range(depth):
            layers.append(ResLayer(num_filts))
        self.feats = nn.Sequential(*layers)

    def forward(self, x, prev_latent=None):
        if self.use_prev_latent and prev_latent is not None:
            prev_latent = prev_latent.to(x.device)
            x = torch.cat([x, prev_latent], dim=1)
        loc_emb = self.feats(x)
        out = self.val_emb(loc_emb)
        return out.squeeze(-1), loc_emb
    

class LinNet(nn.Module):
    def __init__(self, num_inputs):
        super(LinNet, self).__init__()
        self.num_layers = 0
        self.inc_bias = False
        #self.class_emb = nn.Linear(num_inputs, num_classes, bias=self.inc_bias)
        self.val_emb = nn.Linear(num_inputs, 1, bias=self.inc_bias)
        self.feats = nn.Identity()  # does not do anything

    def forward(self, x, class_of_interest=None, return_feats=False):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        #if class_of_interest is None:
        #    class_pred = self.class_emb(loc_emb)
        #else:
        #    class_pred = self.eval_single_class(loc_emb, class_of_interest)
        out = self.val_emb(loc_emb)  # shape: (B, 1)
        return out.sqeeze(-1) #torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return x @ self.class_emb.weight[class_of_interest, :] + self.class_emb.bias[class_of_interest]
        else:
            return x @ self.class_emb.weight[class_of_interest, :]
