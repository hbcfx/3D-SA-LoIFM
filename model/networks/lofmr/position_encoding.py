import math
import torch
from torch import nn

class PositionEncodingSine3D(nn.Module):
    def __init__(self, d_model, max_shape=(256, 256, 256)):
        super(PositionEncodingSine3D, self).__init__()

        pe = torch.zeros((d_model, *max_shape))
        position = torch.arange(0, d_model, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, max_shape, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / max_shape)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 4096, 512)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 3-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256,256), temp_bug_fix=True):
        """
        Args: d_model was the output dim of the backbone
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        x_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        y_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        z_position = torch.ones(max_shape).cumsum(2).float().unsqueeze(0)

        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//3, 2).float() * (-math.log(10000.0) / (d_model//3)))   ##d_model//3  for each type of feature in(sinx,cosx,siny,cosy...)
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//3, 2).float() * (-math.log(10000.0) / d_model//3))
        div_term = div_term[:, None, None,None]  # [C//6, 1, 1,1]
        '''print(pe[0::6, :, :].shape,x_position.shape,div_term.shape)
        print(torch.sin(x_position * div_term).shape)
        print(pe[0::6, :, :].shape)
        print(pe[1::6, :, :].shape)
        print(pe[2::6, :, :].shape)
        print(pe[3::6, :, :].shape)
        print(pe[4::6, :, :].shape)
        print(pe[5::6, :, :].shape)'''
        pe[0::6, :, :, :] = torch.sin(x_position * div_term)  # from 0,step by 6
        pe[1::6, :, :, :] = torch.cos(x_position * div_term)
        pe[2::6, :, :, :] = torch.sin(y_position * div_term)
        pe[3::6, :, :, :] = torch.cos(y_position * div_term)
        pe[4::6, :, :, :] = torch.sin(z_position * div_term)
        pe[5::6, :, :, :] = torch.cos(z_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W,D]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W, D]
        """
        #print(x.shape,self.pe[:, :, :x.size(2), :x.size(3), :x.size(4)].shape)
        return x + self.pe[:, :, :x.size(2), :x.size(3), :x.size(4)]


class PositionEncodingSine2D(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args: d_model was the output dim of the backbone
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)    #from 0,step by 4
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)


        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
