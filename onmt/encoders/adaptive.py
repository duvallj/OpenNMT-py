"""
Implementation of Adaptive Embedding from :cite:https://arxiv.org/abs/1809.10853
Based on https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
"""

import torch
import torch.nn as nn

class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        """
        Args:
            n_token (int): Overall size of the dictionary of tokens to look up
            d_embed (int): Size of the largest embedding dimension. The
                other groups are factors of `div_val` smaller.
            d_proj (int): Size of the output embedding dimension, what all
                embeddings are projected to and concatenated. Usually the
                same as `d_embed`.
            cutoffs (list[int]): The end of each of the groups of tokens
                with common embedding dimensions, not including the final
                group which always ends at `n_token`.
            div_val (int): The factor to reduce each group's embedding
                dimension by.
            sparse (bool): Whether to make our embeddings sparse or not
        Properties:
            n_token, d_embed, d_proj, cutoffs, div_val: same as in args
            emb_layers (nn.ModuleList[nn.Embedding]): All the embeddings
            emb_projs (nn.ModuleList[nn.Linear]): All the projection layers
                to `d_proj`
        """

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.div_val = div_val

        self.emb_scale = d_proj ** 0.5

        self.cutoffs = [0] + cutoffs + [n_token]

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ModuleList()

        if div_val == 1:
            # We just need the one embedding, everything will be the same size
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sparse)
            )
            self.emb_projs.append(
                nn.Linear(d_embed, d_proj, bias=False)
            )
        else:
            for i in range(len(self.cutoffs) - 1):
                start_inc, end_exc = self.cutoffs[i], self.cutoffs[i+1]
                d_embed_i = d_embed // (div_val ** i)
                self.emb_layers.append(
                    nn.Embedding(end_exc - start_inc, d_embed_i, sparse=sparse)
                )
                self.emb_projs.append(
                    nn.Linear(d_embed_i, d_proj, bias=False)
                )

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            embed = self.emb_projs[0](embedding)
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            embed_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                     dtype=param.dtype, device=param.device)

            for i in range(len(self.cutoffs) - 1):
                start_inc, end_exc = self.cutoffs[i], self.cutoffs[i+1]

                # Get all the elements in the input that fall in this range
                mask_i = (inp_flat >= start_inc) & (inp_flat < end_exc)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    # There are no elements, go to the next group
                    continue

                # Convert the overall indicies into indices for this embedding
                inp_i = inp_flat.index_select(0, indices_i) - start_inc

                # Get the corresponding embedding
                embed_i = self.emb_layers[i](inp_i)
                embed_i = self.emb_projs[i](embed_i)

                # Copy back to the main embedding array
                embed_flat.index_copy_(0, indices_i, embed_i)

            embed = embed_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed

