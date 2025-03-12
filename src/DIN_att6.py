
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, DIN_Attention, Dice
from fuxictr.utils import not_in_whitelist



class target_attention(nn.Module):
    def __init__(self, 
                 embedding_dim=64,
                 num_heads=4,
                 dropout=0.1,
                 use_softmax=True):
        super(target_attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.use_softmax = use_softmax
        
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, target_item, history_sequence, mask=None):
        batch_size, seq_len, _ = history_sequence.size()
        
        query = self.query_proj(target_item)
        if len(target_item.shape) == 2:
            query = query.unsqueeze(0)
        elif len(target_item.shape) == 3:
            query = query.transpose(0, 1)
        key = self.key_proj(history_sequence.transpose(0, 1))
        value = self.value_proj(history_sequence.transpose(0, 1)) 
 
        if mask is not None:
            attn_mask = (~mask.bool())
        else:
            attn_mask = None
        
        # Cross-attention
        attn_output, _ = self.multihead_attn(query=query,
                                             key=key,
                                             value=value,
                                             key_padding_mask=attn_mask)
        
        if attn_output.shape[0] ==1:
             attn_output = attn_output.squeeze(0)
        else:
            attn_output = attn_output.transpose(0, 1)
        
        return attn_output
    
class RMSNorm(nn.Module):
  def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_size))
  
  def _norm(self, hidden_states: Tensor) -> Tensor:
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + self.eps)
  
  def forward(self, hidden_states: Tensor) -> Tensor:
    return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)

class CrossAttention_Layer(nn.Module):
    # 将layernorm改为RMSNorm
    def __init__(self, 
                 embedding_dim=256,
                 num_heads=4,
                 ff_dim=256,
                 dropout=0.1,
                 use_softmax=True):
        super(CrossAttention_Layer, self).__init__()
        self.embedding_dim = embedding_dim

        self.cross_attn = target_attention(embedding_dim, num_heads, dropout, use_softmax)
        self.attn_norm = RMSNorm(embedding_dim) # RMSNorm
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_item, history_sequence, mask=None): 
        history_sequence = self.attn_norm(history_sequence)
        attn_out = self.cross_attn(target_item, history_sequence, mask)
        attn_out = self.attn_norm(target_item + self.dropout(attn_out)) # 残差连接
        
        ffn_out = self.ffn(attn_out)
        output = attn_out + self.dropout(ffn_out)
        
        return output

class TargetAttentionEncoder(nn.Module):
    # 将layernorm改为RMSNorm
    def __init__(self,
                 num_layers=2,
                 embedding_dim=256,
                 num_heads=4,
                 ff_dim=256,
                 dropout=0.1,
                 use_softmax=True,
                 if_use_self_attn= False):
        super(TargetAttentionEncoder, self).__init__()
        self.if_use_self_attn = if_use_self_attn
        if self.if_use_self_attn:
            self.selfattention_layer = CrossAttention_Layer(embedding_dim, num_heads, ff_dim, dropout, use_softmax)
        self.layers = nn.ModuleList([
            CrossAttention_Layer(embedding_dim, num_heads, ff_dim, dropout, use_softmax)
            for _ in range(num_layers)
        ])
        

    def forward(self, target_item, history_sequence, mask=None):
        output = target_item
        if self.if_use_self_attn:
            history_sequence = self.selfattention_layer(history_sequence, history_sequence, mask)
        for i,layer in enumerate(self.layers):
            output = layer(output, history_sequence, mask)
        return output



class DIN_att6(BaseModel):
    # 目前最佳，加入了target与user_seq的交叉注意力机制，加入了user_id的embedding，增加了dnn的层数
    # likes_level和likes_num应该也属于item的特征
    def __init__(self, 
                 feature_map, 
                 model_id="DIN_att5", 
                 gpu=0, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=64, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_use_softmax=False,
                 accumulation_steps=1,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 if_use_self_attn=False,
                 num_layers = 2,
                 **kwargs):
        super(DIN_att6, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim # 64
        self.item_info_dim = 0  # 所有source为item的特征的embedding_dim之和 64+64+128=256,
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)
        ##更改了 
        self.item_info_dim += embedding_dim*2
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.attention_layers = DIN_Attention(
                                            self.item_info_dim,
                                            attention_units=attention_hidden_units,
                                            hidden_activations=attention_hidden_activations,
                                            output_activation=attention_output_activation,
                                            dropout_rate=attention_dropout,
                                            use_softmax=din_use_softmax)
 
        self.target_attention = TargetAttentionEncoder(num_layers = num_layers,
                                                    embedding_dim = self.item_info_dim,
                                                     num_heads=4,
                                                     ff_dim=self.item_info_dim,
                                                     dropout=attention_dropout,
                                                     use_softmax=True,
                                                     if_use_self_attn = if_use_self_attn)

        input_dim = embedding_dim *3+ self.item_info_dim*3    # 
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        '''
        inputs:
            batch_dict = {user_id: [batch_size],
                      likes_level: [batch_size],
                      view_level: [batch_size] }
            item_dict = {item_id: [batch_size*(64+1)],
                     item_tags: [batch_size*(64+1),5],
                     item_emb_d128: [batch_size*(64+1),128],
                     likes_level: [batch_size*65],
                     view_level: [batch_size*65],}
            mask = [batch_size, 64]  
        '''
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict: # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)   # torch.Size([batch_size, 64*3]) likes_level, view_level, user_id
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)  
        batch_size = mask.shape[0]   # mask: [batch_size, 64]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)  # torch.Size([batch_size, 65, 128+64*4])
        target_emb = item_feat_emb[:, -1, :]  # torch.Size([batch_size, 128+64*4])
        sequence_emb = item_feat_emb[:, 0:-1, :]  # torch.Size([batch_size, 64, 128+64*4])
        
        pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)
        
        pooling_emb_cross = self.target_attention(target_emb, sequence_emb, mask)
        
        emb_list += [target_emb, pooling_emb, pooling_emb_cross]  # [feature_emb, target_emb, pooling_emb,pooling_emb_cross] 
        feature_emb = torch.cat(emb_list, dim=-1) # torch.Size([batch_size, self.item_info_dim*3+embe_dim*3])
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs

        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            # if feature_spec["type"] == "meta":
            #     continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss
