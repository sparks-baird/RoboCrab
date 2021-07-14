import numpy as np
import pandas as pd

import torch
from torch import nn


# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


# %%
class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = "data/element_properties"
        # Choose what element information the model receives
        mat2vec = f"{elem_dir}/mat2vec.csv"  # element embedding
        # mat2vec = f'{elem_dir}/onehot.csv'  # onehot encoding (atomic number)
        # mat2vec = f'{elem_dir}/random_200.csv'  # random vec for elements

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))

        # e.g. size: (118 elements + 1, 200 features)
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(
            self.compute_device, dtype=data_type_torch
        )

        # e.g. size: ()
        sbfv = np.ones_like(cbfv)
        feat_size = sbfv.shape[-1]
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, sbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.sbfv = nn.Embedding.from_pretrained(cat_array).to(
            self.compute_device, dtype=data_type_torch
        )

    def forward(self, src, cat_feat, bool_src, float_feat):
        """
        Compute elemental and structural embedding using elemental indices and robocrystallographer features.

        Parameters
        ----------
        src : torch.Tensor (Batch Size, # elements)
            Elemental indices (padded). E.g. hydrogen encoded as 1.
        cat_feat : torch.Tensor (Batch Size, # features = 3)
            Categorical robocrystallographer features. E.g. 'dimensionality'.
        bool_src : torch.Tensor (Batch Size, # features = 15)
            Boolean robocrystallographer feature indices (padded). E.g. 'contains_tetrahedral'.
        float_feat : torch.Tensor (Batch Size, # features = 44)
            Numerical robocrystallographer features. E.g. 'average_bond_length'.

        Returns
        -------
        x_emb : torch.Tensor (Batch Size, # features = # Elements + cat_feat.shape[2] + bool_feat.shape[2] + float_feat.shape[2], 200)
            Embedding matrix (post FC-network) for mat2vec and robocrystallographer features (vertically stacked).

        """
        mat2vec_emb = self.cbfv(src)

        bool_emb = self.sbfv(bool_src)

        # stack mat2vec_emb and (expanded/repeated) structural features
        feats = [cat_feat, float_feat]
        d = [1, 1, mat2vec_emb.shape[2]]
        cat_feat, float_feat = [feat.unsqueeze(2).repeat(d) for feat in feats]

        # size e.g. (256, # Elements + # Structural features = 159, 200)
        feats = torch.cat([mat2vec_emb, cat_feat, bool_emb, float_feat], dim=1)

        # size e.g. (256, 159, 512)
        x_emb = self.fc_mat2vec(feats)
        return x_emb

        """mini code graveyard"""
        """
                # to determine filler dimension
        # bool_len = list(map(len, bool_src))
        # mx = max(bool_len)  # this might need to be defined earlier for the full dataset

        # add filler zeros
        # bool_src = [
        #     torch.cat(
        #         [
        #             bools,
        #             torch.zeros(
        #                 mx - len(bools), dtype=bool, device=self.compute_device
        #             ),
        #         ]
        #     )
        #     for bools in bool_src
        # ]
        # bool_src = pad(bools[[0, 0], [0, mx - len(bools)]])
        # bool_src = torch.stack(bool_src)
        
        # feats = torch.cat(feats, dim=1)
        
        # cat_feat.repeat([1, len(mat2vec_emb)], 1)
        """


# %%
class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired by the positional encoder discussed by Vaswani.
    
    See https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(
            0, self.resolution - 1, self.resolution, requires_grad=False
        ).view(self.resolution, 1)
        fraction = (
            torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False)
            .view(1, self.d_model)
            .repeat(self.resolution, 1)
        )

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x)) ** 2
            x[x > 1] = 1
            # x = 1 - x  # for sinusoidal encoding at x=0
        x[x < 1 / self.resolution] = 1 / self.resolution
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out


# %%
class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.0]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.0]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.0]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                self.d_model, nhead=self.heads, dim_feedforward=2048, dropout=0.1
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.N
            )

    def forward(self, src, frac, cat_feat, bool_src, float_feat):
        # scaled, fully-connected mat2vec (fc_mat2vec) embedding, see Fig 6 of 10.1038/s41524-021-00545-1
        x = self.embed(src, cat_feat, bool_src, float_feat) * 2 ** self.emb_scaler

        nrobo_feat = x.shape[1] - src.shape[1]

        # "fractional coordinates" for structural features are constant (ones or scaled constant)
        # should I divide by the number of structural features? Probably best to have some type of normalization to avoid nans
        # normalization possibly should be the # of True's in each compound?
        d = [frac.shape[0], nrobo_feat]
        ones = torch.ones(d, device=self.compute_device, dtype=src.dtype)
        frac = torch.cat([frac, ones / nrobo_feat], dim=1)

        # mask has 1 if n-th element is present, 0 if not. E.g. single element compound has mostly mask of 0's
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        # fractional encoding, see Fig 6 of 10.1038/s41524-021-00545-1
        pe = torch.zeros_like(x)  # prevalence encoding
        ple = torch.zeros_like(x)  # prevalence log encoding
        pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2

        # first half of features are prevalence encoded (i.e. 512//2==256)
        pe[:, :, : self.d_model // 2] = self.pe(frac) * pe_scaler
        # second half of features are prevalence log encoded
        ple[:, :, self.d_model // 2 :] = self.ple(frac) * ple_scaler

        if self.attention:
            # sum of fc_mat2vec embedding (x), prevalence encoding (pe), and prevalence log encoding (ple)
            # see Fig 6 of 10.1038/s41524-021-00545-1
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)

            # transformer encoding
            """note on src_key_padding_mask: if provided, specified padding elements
            in the key will be ignored by the attention. When given a binary mask
            and a value is True, the corresponding value on the attention layer
            will be ignored. When given a byte mask and a value is non-zero, the
            corresponding value on the attention layer will be ignored.
            Source: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html"""
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        """0:1 index eliminates the repeated values (down to 1 colummn)
        repeat() fills it back up (to e.g. d_model == 512 values)"""
        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            # set values of x which correspond to an element not being present to 0
            x = x.masked_fill(hmask == 0, 0)

        return x

        """mini code graveyard"""
        """
        #nrobo_feat = sum([feat.shape[1] for feat in [cat_feat, bool_feat, float_feat]])
        """


# %%
class CrabNet(nn.Module):
    """
    Compositionally restricted attention based network for predicting material properties.
    
    keep out_dims set to 3 (important for having correct shape for prediction/uncertainty)
    """

    def __init__(self, out_dims=3, d_model=512, N=3, heads=4, compute_device=None):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.encoder = Encoder(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            compute_device=self.compute_device,
        )
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

    def forward(self, src, frac, cat_feat, bool_src, float_feat):
        """
        

        Parameters
        ----------
        src : TYPE
            DESCRIPTION.
        frac : TYPE
            DESCRIPTION.
        cat_feat : TYPE
            DESCRIPTION.
        bool_src : TYPE
            DESCRIPTION.
        float_feat : TYPE
            DESCRIPTION.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        # e.g. size: (256 batch size, 3 elements, 512 features)
        output = self.encoder(src, frac, cat_feat, bool_src, float_feat)
        batch_size = output.shape[0]

        # elemental mask (so you only average "elements", not the filler dimensions)
        emask = src == 0

        # structural masks setup
        nrobo_feats = [feat.shape[1] for feat in [cat_feat, bool_src, float_feat]]
        d1, d2, d3 = [[batch_size, n] for n in nrobo_feats]  # d2 unused

        # categorical mask
        cmask = torch.zeros(d1, device=self.compute_device, dtype=bool)

        # boolean mask
        bmask = bool_src == 0

        # float mask
        fmask = torch.zeros(d3, device=self.compute_device, dtype=bool)

        # concatenate the various masks
        mask = torch.cat([emask, cmask, bmask, fmask], dim=1)

        # massage/duplicate into correct size, e.g. size (32 batch_size, 159 elemental+structural features, 3 out_dims)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.out_dims)

        # simple linear, e.g. size (32 batch_size, 159 elemental+structural features, 3 out_dims)
        output = self.output_nn(output)

        # average the "element contribution" at the end e.g. size (32 batch size, 3 elements)
        if self.avg:
            # fill True locations of mask with 0
            output = output.masked_fill(mask, 0)

            #
            output = output.sum(dim=1) / (~mask).sum(dim=1)

            # e.g. size (32 batch_size, 3 elements)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, : logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output

    """mini code graveyard"""
    """
    # boolean mask (nice that we can simply invert the original boolean feature matrix)
    # bmask = ~bool_feat
    """


# %%
if __name__ == "__main__":
    model = CrabNet()
