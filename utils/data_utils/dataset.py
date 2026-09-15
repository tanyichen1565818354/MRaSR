# utils/data_utils/dataset.py
import torch
import os
from torch.utils.data import Dataset

class RecSysDataset(Dataset):
    def __init__(self, data_path: str, max_seq_len: int = 50, is_test: bool = False):
        self.sequences = torch.load(data_path, weights_only=False)  # 加载预处理后的序列数据
        self.max_seq_len = max_seq_len
        self.is_test = is_test
        self.user_ids = list(self.sequences.keys())

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_data = self.sequences[user_id]
        
        if self.is_test and 'input_sequence' in user_data:
            # 验证集/测试集使用留一法评估
            input_seq = user_data['input_sequence']
            target = user_data['target_item']
            
            # 序列截断与填充
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]
                
            return {
                'user_id': user_id,
                'input_ids': torch.LongTensor(input_seq),
                'target': torch.LongTensor([target])
            }
        else:
            # 训练集使用下一项预测
            seq = user_data['sequence']
            
            # 序列截断与填充
            if len(seq) > self.max_seq_len + 1:
                seq = seq[-(self.max_seq_len+1):]
                
            return {
                'user_id': user_id,
                'input_ids': torch.LongTensor(seq[:-1]),  # 输入序列
                'labels': torch.LongTensor(seq[1:])       # 目标序列
            }

class TrainDataset(RecSysDataset):
    def __init__(self, data_dir: str, max_seq_len: int = 50):
        super().__init__(os.path.join(data_dir, "sequences.pth"), max_seq_len, is_test=False)

class ValidDataset(RecSysDataset):
    def __init__(self, data_dir: str, max_seq_len: int = 50):
        super().__init__(os.path.join(data_dir, "sequences.pth"), max_seq_len, is_test=True)

class TestDataset(RecSysDataset):
    def __init__(self, data_dir: str, max_seq_len: int = 50):
        super().__init__(os.path.join(data_dir, "sequences.pth"), max_seq_len, is_test=True)
