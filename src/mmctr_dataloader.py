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

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import polars as pl
import torch


class ParquetDataset(Dataset):
    def __init__(self, data_path):
        self.column_index = dict()
        self.darray = self.load_data(data_path)
        
    def __getitem__(self, index):
        return self.darray[index, :]
    
    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        data_arrays = []
        idx = 0
        for col in df.columns:
            if df[col].dtype == "object":
                array = np.array(df[col].to_list())
                seq_len = array.shape[1]
                self.column_index[col] = [i + idx for i in range(seq_len)]
                idx += seq_len
            else:
                array = df[col].to_numpy()
                self.column_index[col] = idx
                idx += 1
            data_arrays.append(array)
        return np.column_stack(data_arrays)


class MMCTRDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, item_info, batch_size=32, shuffle=False,
                 num_workers=1, max_len=100, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(data_path)
        column_index = self.dataset.column_index
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map, max_len, column_index, item_info))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches
    

class BatchCollator(object):
    def __init__(self, feature_map, max_len, column_index, item_info,):
        self.feature_map = feature_map
        self.item_info = pd.read_parquet(item_info)
        item_like_vilel_level= pd.read_parquet('./data/item_feature.parquet',columns=["item_id", "likes_level", "views_level"])

        item_0 = pd.DataFrame({
        "item_id": 0,
        "likes_level": 0,   
        "views_level": 0  
        }, index=[0])
        self.item_like_vilel_level  = pd.concat([item_0, item_like_vilel_level],ignore_index=True)
        # self.item_feature_info=pd.read_parquet('./data/MicroLens_1M_x1/item_info_new.parquet')
        self.max_len = max_len
        self.column_index = column_index
    
    
    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = set(list(self.feature_map.features.keys()) + self.feature_map.labels)
        batch_dict = dict()
        for col, idx in self.column_index.items():
            if col in all_cols:
                batch_dict[col] = batch_tensor[:, idx]
        batch_seqs = batch_dict["item_seq"][:, -self.max_len:]
        del batch_dict["item_seq"]   # 删除了user端batch_dict里的"item_seq"，只保留了batch_seqs
        mask = (batch_seqs > 0).float() # zeros for masked positions
        item_index = batch_dict["item_id"].numpy().reshape(-1, 1)
        del batch_dict["item_id"]   # 删除了user端batch_dict里的"item_id"，实际用户历史交互的item_id和候选物品的item_id集成到了batch_items里
        batch_items = np.hstack([batch_seqs.numpy(), item_index]).flatten() # [batch_size,64] + [batch_size ,1] -> [batch_size,65]-> [batch_size*65]
        item_info = self.item_info.iloc[batch_items]   # 提取user的item_seq和候选item_id,fatten()为1维，然后再将其作为item_id映射其对应的item_tags与item_emb
        item_dict = dict()  
        
        item_likes_views = self.item_like_vilel_level.iloc[batch_items]  # user_id 的likes_level和views_level
        
        for col in item_info.columns:
            if col in all_cols:
                item_dict[col] = torch.from_numpy(np.array(item_info[col].to_list()))
    
        for col in ["likes_level","views_level"]:    # 构建历史序列的likes_level和views_level
            item_dict[col] = torch.from_numpy(np.array(item_likes_views[col].to_list()))
    
    
        return batch_dict, item_dict, mask
    
    
   
    