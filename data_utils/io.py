import json
import numpy as np
from monai.data import NibabelWriter
from monai.transforms import EnsureChannelFirst


def save_json(data, file_path, sort_keys=True):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=sort_keys)
    print(f'save json to {file_path}')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f'load json from {file_path}')
        return data


def save_img(img, img_meta_dict, pth):
    writer = NibabelWriter()
    # --- 關鍵修正：手動增加通道維度 ---
    # 我們知道 img 是一個 (H, W, D) 的 NumPy 陣列
    # 我們使用 np.expand_dims 在最前面 (axis=0) 增加一個通道維度
    # 使其變為 (1, H, W, D)
    img_with_channel = np.expand_dims(img, axis=0)
    
    writer.set_data_array(img_with_channel)
    # ------------------------------------
    writer.set_metadata(img_meta_dict)
    writer.write(pth, verbose=True)
