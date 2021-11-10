""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul
# from onmt.utils.misc import aeq


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, attn_type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and attn_type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return multi-head attn
        attns = attn \
            .view(batch_size, head_count,
                  query_len, key_len)

        return output, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout

class RelMultiHeadedAttention(nn.Module):
    """Relative Multi-Headed Attention module from Transformer-XL
    :cite:https://arxiv.org/pdf/1901.02860v3.pdf

    Similar to standard multi-headed attention, but uses relative encodings
    when computing key vectors, and trainable parameters to replace queries
    that used to be absolutely encoded.

    This is different than the relative encodings enabled by setting
    `max_relative_positions` in the standard MultiHeadedAttention module.

    Heavily drawn from the [author's published code](https://github.com/kimiyoung/transformer-xl),
    specifically their `RelPartialLearnableMultiHeadAttn` module that they
    present in the paper.

    Args:
       head_count (int): number of parallel heads
       head_dim (int): the dimension of each head
       model_dim (int): the dimension of keys/values/queries. Ideally this is
                        head_count * head_dim
       dropatt (float): dropout parameter after computing attention,
           NOTE: This is the same as `dropout` on MultiHeadedAttention
       dropout (float): dropout parameter after final linear layer
       pre_lnorm (bool): whether to do layer normalization on input/output
    """
    def __init__(self, head_count, model_dim, dropatt=0.0, dropout=0.1, pre_lnorm=False):
        super(RelMultiHeadedAttention, self).__init__()

        self.model_dim = model_dim
        self.head_dim = head_dim
        self.head_count = head_count

        self.linear_qkv = nn.Linear(model_dim, 3 * head_count * head_dim,
                                    bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.dropatt = nn.Dropout(dropatt)
        self.dropout = nn.Dropout(dropout)
        self.linear_relative = nn.Linear(model_dim, head_count * head_dim,
                                         bias=False)
        self.linear_output = nn.Linear(head_count * head_dim, model_dim,
                                       bias=False)

        self.pre_lnorm = pre_lnorm
        if pre_lnorm: self.layer_norm = nn.LayerNorm(model_dim)

        self.scale = 1 / (self.dim_per_head ** 0.5)

    def _rel_shift(self, x):
        """
        Args:
            x (Tensor(height, width, ...))
        Returns:
            A version of x where the last row remains in place, the second to
            last is left-shifted by one, the third to last is left-shifted by
            two, and so on. The rationale is presented in Appendix B of
            :cite:https://arxiv.org/pdf/1901.02860v3.pdf
        """
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask, mems=None):
        """
        Args:
            w (Tensor(query_len, batch_size, model_dim)):
                The output from the previous layer
            r (Tensor(relative_len, batch_size, model_dim)):
                Tensor representing the relative position encodings
            r_w_bias (Tensor(head_count, head_dim)):
                Learned bias for term (c), that is the query vectors against
                keys given by the relative output of the previous layer
            r_r_bias (Tensor(head_count, head_dim)):
                Learned bias for term (d), that is the query vectors against
                keys given by the relative position
            attn_mask (Tensor(query_len, key_len) or
                       Tensor(query_len, key_len, batch_size)):
                How to mask the resulting attention so that non-desired
                shifted parts of the relative matrix don't interfere with our
                calculations.
            mems (Tensor(memory_len, batch_size, model_dim)):
                The previous context vectors to concatenate to our current one
        Returns:
            Tensor(query_len, batch_size, model_dim): The context vector for
            the next layer
        """
        qlen, rlen, batch_size = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            w = torch.cat([mems, w], 0)
            if self.pre_lnorm: w = self.layer_norm(w)

            w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm: w = self.layer_norm(w)

            w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        r_head_k = self.linear_relative(r)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, batch_size, self.head_count, self.head_dim)
        w_head_k = w_head_q.view(klen, batch_size, self.head_count, self.head_dim)
        w_head_v = w_head_q.view(klen, batch_size, self.head_count, self.head_dim)
        r_head_k = r_head_k.view(rlen, self.head_count, self.head_dim)

        rw_head_q = w_head_q + r_w_bias
        # (qlen, klen, batch_size, head_count)
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_r_bias
        # (qlen, klen, batch_size, head_count)
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)

        # (qlen, klen, batch_size, head_count)
        attn_scores = AC + BD
        attn_scores.mul_(self.scale)

        if attn_mask.dim() == 2:
            attn_scores = attn_scores.float().masked_fill(
                attn_mask[:,:,None,None], -float('inf')
            ).type_as(attn_scores)
        elif attn_mask.dim() == 3:
            attn_scores = attn_scores.float().masked_fill(
                attn_mask[:,:,:,None], -float('inf')
            ).type_as(attn_scores)
        else:
            raise RuntimeError(f"Dimension error: attn_mask should have 2 or 3 dimensions, not {attn_mask.dim()}")

        attn_probs = self.softmax(attn_scores)
        attn_probs = self.dropatt(attn_probs)

        # (qlen, batch_size, head_count, head_dim)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_probs, w_head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0),
            attn_vec.size(1),
            self.head_count * self.head_dim
        )

        attn_out = self.linear_output(attn_vec)
        attn_out = self.dropout(attn_out)

        # Make a residual connection
        output = w + attn_out
        if self.pre_lnorm: output = self.layer_norm(output)

        return output
