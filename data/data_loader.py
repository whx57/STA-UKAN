import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split,Subset
from typing import Tuple, Dict
import scipy.ndimage
import netCDF4 as nc
from scipy.interpolate import griddata
import torch
import os
import sys
sys.path.append("/home/nxd/wx/wx/downscale_and_bias_correction/")

class CustomDataset(Dataset):
    def __init__(self, X, Y, MASK, bzh: bool = False):
        """
        bzh == True  →  对 y 做 (y-mean)/std 标准化
        bzh == False →  保留原始量纲
        """
        self.X, self.Y, self.MASK = X, Y, MASK
        self.bzh = bzh
        if self.bzh:                               # 只在需要时计算 μ,σ
            self.y_mean = Y.mean()
            self.y_std  = Y.std()
        # print(MASK.shape, X.shape, Y.shape)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y = self.Y[idx]
        
        if self.bzh:
            # print("on gyh")
            y = (y - self.y_mean) / self.y_std
        return self.X[idx], y, self.MASK[0]    # 保持原返回签名

def load_data(x_path, y_path, m_path):

    X = np.load(x_path)
    Y = np.load(y_path)
    MASK = np.load(m_path)
    MASK = np.repeat(MASK, 2, axis=0)
    MASK = torch.tensor(MASK, dtype=torch.bool)
    if MASK.dim() == 4:
        # 形如 [1, 128, H, W] → [1, H, W]
        if MASK.shape[1] > 1:
            MASK = MASK[:, 0, :, :]
        else:
            MASK = MASK.squeeze(1)  # [1,1,H,W] → [1,H,W]

    if MASK.dim() == 3 and MASK.shape[0] > 1:
        MASK = MASK[0, :, :]      # 多个通道的 → 只取第一个变量

    if MASK.dim() == 2:
        MASK = MASK.unsqueeze(0)  # [H,W] → [1,H,W]

    return X, Y, MASK

def prepare_dataloaders(X, Y, MASK, batch_size,map_path,split_mode='random', train_sets=None,  seasons=None ,bzh=False ):

    topo = load_and_preprocess_map(map_path)
    map_repeated = prepare_map(topo, X.shape[0])
    X_augmented = np.concatenate((X, map_repeated), axis=1)
    X_torch = torch.tensor(X_augmented, dtype=torch.float32)
    y_torch = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    return create_dataloaders(X_torch, y_torch, MASK, batch_size,
                              split_mode=split_mode,
                              train_sets=train_sets,seasons=seasons,bzh=bzh) 

def load_and_preprocess_map(map_path):
    print('-------------',map_path)
    # data_map = nc.Dataset(map_path, 'r', format='NETCDF4')
    # missing_value = -9.0E33
    # topo = np.array(data_map.variables['topo'][:])
    # topo = np.flip(topo, axis=0)
    # topo[topo == missing_value] = np.nan
    # topo_interpolated = interpolate_missing_data(topo)
    topo_interpolated = np.load(map_path, allow_pickle=True)
    resampled_topo = resize_data(topo_interpolated, target_shape=(32, 48))
    mean = np.nanmean(resampled_topo)
    std = np.nanstd(resampled_topo)
    resampled_topo_normalized = (resampled_topo - mean) / std
    return resampled_topo_normalized

def interpolate_missing_data(topo):
    x = np.arange(0, topo.shape[1])
    y = np.arange(0, topo.shape[0])
    x_grid, y_grid = np.meshgrid(x, y)
    mask = np.isnan(topo)
    known_values = topo[~mask]
    return griddata((x_grid[~mask], y_grid[~mask]), known_values, (x_grid, y_grid), method='nearest')

def resize_data(data, target_shape):
    scale_x = target_shape[1] / data.shape[1]
    scale_y = target_shape[0] / data.shape[0]
    return scipy.ndimage.zoom(data, (scale_y, scale_x), order=1)

def prepare_map(map_data, num_samples):
    map_expanded = np.expand_dims(np.expand_dims(map_data, axis=0), axis=0)
    return np.repeat(map_expanded, num_samples, axis=0)
def create_dataloaders(X_torch: torch.Tensor, y_torch: torch.Tensor, MASK: torch.Tensor,
                      batch_size: int, split_mode: str = 'random', seed: int = 42,train_sets=None, seasons=None,bzh=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits the dataset into training, validation, and testing sets based on the specified split mode.

    Args:
        X_torch (torch.Tensor): Input features tensor.
        y_torch (torch.Tensor): Target features tensor.
        MASK (torch.Tensor): Mask tensor.
        batch_size (int): Batch size for DataLoaders.
        split_mode (str): 'random', 'sequential', or 'year' for splitting.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """
    # Set the random seed for reproducibility
    # print('seed:', seed)
    # torch.manual_seed(seed)
    
    # Initialize the custom dataset
    dataset = CustomDataset(X_torch, y_torch, MASK,bzh)
    total_size = len(dataset)

    # 若未指定则默认四季全用
    if split_mode == 'season':
        # 若未指定则默认四季全用
        if seasons is None:
            seasons = ['spring','summer','autumn','winter']
        allowed = set()
        for s in seasons:
            allowed.update(SEASON_INDICES[s])

        season_idx_sorted = sorted(allowed)          # 时间顺序
        season_subset     = Subset(dataset, season_idx_sorted)

        total = len(season_subset)
        tr = int(0.8 * total)
        va = int(0.1 * total)
        te = total - tr - va
        g  = torch.Generator().manual_seed(seed)
        train_idx     = season_idx_sorted[:tr]
        validate_idx  = season_idx_sorted[tr:tr+va]
        test_idx      = season_idx_sorted[tr+va:]
        train_dataset      = Subset(dataset, train_idx)
        validate_dataset        = Subset(dataset, validate_idx)
        test_dataset       = Subset(dataset, test_idx)
        # train_dataset, validate_dataset, test_dataset = random_split(season_subset, [tr, va, te], generator=g)
        train_shuffle = True


    elif split_mode == 'random':
    # Define split sizes
        train_size = int(0.8 * total_size)
        validate_size = int(0.1 * total_size)
        test_size = total_size - train_size - validate_size
        
        # Randomly split the dataset
        generator = torch.Generator().manual_seed(seed)
        train_dataset, validate_dataset, test_dataset = random_split(
            dataset, [train_size, validate_size, test_size], generator=generator)
        train_shuffle = True
    
    elif split_mode == 'sequential':
        # Define split sizes
        train_size = int(0.8 * total_size)
        validate_size = int(0.1 * total_size)
        test_size = total_size - train_size - validate_size
        
        # Sequentially split the dataset without shuffling
        indices = list(range(total_size))
        train_indices = indices[:train_size]
        validate_indices = indices[train_size:train_size + validate_size]
        test_indices = indices[train_size + validate_size:]
        
        train_dataset = Subset(dataset, train_indices)
        validate_dataset = Subset(dataset, validate_indices)
        test_dataset = Subset(dataset, test_indices)
        train_shuffle = True
    
    elif split_mode == 'year':
        # Define year-wise index ranges (start and end inclusive)
        # Format: (start_index, end_index)
        year_indices = {
            '06': (0, 103),
            '07': (104, 309),
            '08': (310, 619),
            '09': (620, 929),
            '10': (930, 1239),
            '11': (1240, 1549),
            '12': (1550, 1859),
            '13': (1860, 2169),
            '14': (2170, 2479),
            '15': (2480, 2789),
            '16': (2790, 3099),
            '17': (3100, 3409),
            '18': (3410, 3719),
            '19': (3720, 4029),
            '20': (4030, 4339),
            '21': (4340, 4545),
            '22': (4546, 4649)
        }
        
        # Define which years go into which split
        training_years = [str(year).zfill(2) for year in range(6, 20)]   # '06' to '19'
        validation_year = '20'
        test_years = ['21', '22']
        
        # Collect indices for each split
        train_indices = []
        for year in training_years:
            start, end = year_indices.get(year, (None, None))
            if start is not None and end is not None:
                train_indices.extend(range(start, end + 1))
        
        validate_indices = []
        val_start, val_end = year_indices.get(validation_year, (None, None))
        if val_start is not None and val_end is not None:
            validate_indices.extend(range(val_start, val_end + 1))
        
        test_indices = []
        for year in test_years:
            start, end = year_indices.get(year, (None, None))
            if start is not None and end is not None:
                test_indices.extend(range(start, end + 1))
            if train_sets:
                allowed = set()
                for ds in train_sets:
                    allowed.update(ALL_DS_INDICES[ds])
                train_indices = [i for i in train_indices if i in allowed]
        # Create subsets
        train_dataset = Subset(dataset, train_indices)
        validate_dataset = Subset(dataset, validate_indices)
        test_dataset = Subset(dataset, test_indices)
        train_shuffle = True  
    
    else:
        raise ValueError("split_mode must be 'random', 'sequential', or 'year'")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_shuffle, 
        num_workers=4, pin_memory=True
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    if split_mode in ['random', 'sequential']:
        # Save dataset indices for reproducibility (optional)
        save_dataset_indices(train_dataset, f'train_dataset_{split_mode}_{seed}.pth')
        save_dataset_indices(validate_dataset, f'validate_dataset_{split_mode}_{seed}.pth')
        save_dataset_indices(test_dataset, f'test_dataset_{split_mode}_{seed}.pth')
    elif split_mode == 'year':
        # Save dataset indices for reproducibility (optional)
        save_dataset_indices(train_dataset, f'train_dataset_year_{seed}.pth')
        save_dataset_indices(validate_dataset, f'validate_dataset_year_{seed}.pth')
        save_dataset_indices(test_dataset, f'test_dataset_year_{seed}.pth')
    
    return train_loader, validate_loader, test_loader

def save_dataset_indices(dataset, filename):
    """ Save dataset indices to a file for reproducibility. """
    # indices = dataset.indices
    # torch.save(indices, os.path.join('output', filename))


# ─── 年块在 4650 大数组中的起止行 ───
YEAR_IDX = {
    '06': (0,103),'07':(104,309),'08':(310,619),'09':(620,929),'10':(930,1239),
    '11':(1240,1549),'12':(1550,1859),'13':(1860,2169),'14':(2170,2479),
    '15':(2480,2789),'16':(2790,3099),'17':(3100,3409),'18':(3410,3719),
    '19':(3720,4029),'20':(4030,4339),'21':(4340,4545),'22':(4546,4649)
}

# 每年 3 套数据应出现的计数
YEAR_COUNTS = {
    '06': {21:104},
    '07': {21:104, 22:102},
    # 08‑20：三套齐全
    **{f"{y:02d}": {21:104, 22:102, 23:104} for y in range(8,21)},
    '21': {22:102, 23:104},
    '22': {23:104}
}

# 完整的拼接顺序规则
PREF = {
    '06':[21],
    '07':[21,22],
    # 08‑20：21→23→22 轮转
    **{f"{y:02d}":[21,23,22] for y in range(8,21)},
    '21':[23,22],
    '22':[23]
}

MONTH_TO_SEASON = {
    1:'winter', 2:'winter',
    3:'spring', 4:'spring', 5:'spring',
    6:'summer', 7:'summer', 8:'summer',
    9:'autumn',10:'autumn',11:'autumn',
    12:'winter'
}
SEASON_INDICES = {k: [] for k in ['spring','summer','autumn','winter']}

for yr, (s, e) in YEAR_IDX.items():          # YEAR_IDX 已在旧代码中定义
    n = e - s + 1                            # 当年样本数
    q, r = divmod(n, 12)                     # 基本份额 & 余数
    # 让前 r 个月多 1 个样本，保证尽量平均
    month_sizes = [q + 1 if i < r else q for i in range(12)]
    idx = s
    for m, sz in enumerate(month_sizes, 1):
        SEASON_INDICES[MONTH_TO_SEASON[m]].extend(range(idx, idx + sz))
        idx += sz

def _year_membership(year:str):
    """返回该年份块中 4650 索引 -> 数据集 ID(21/22/23) 的映射列表"""
    s, e          = YEAR_IDX[year]
    counts_left   = YEAR_COUNTS[year].copy()     # 剩余配额
    order         = PREF[year]                   # 轮转顺序
    membership    = []
    ptr           = 0
    for _ in range(e-s+1):
        # 找到下一个还有配额的 data‑set
        while counts_left.get(order[ptr % len(order)], 0) == 0:
            ptr += 1
        ds_id = order[ptr % len(order)]
        membership.append(ds_id)
        counts_left[ds_id] -= 1
        ptr += 1
    return membership            # 长度 = 年度样本数
ALL_DS_INDICES = {21: [], 22: [], 23: []}
for yr, (s, e) in YEAR_IDX.items():
    mem = _year_membership(yr)           # 年度 membership
    for local_i, ds in enumerate(mem):
        ALL_DS_INDICES[ds].append(s + local_i)