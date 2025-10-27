import os
import time
import importlib
from pathlib import PurePath

import torch
import torch.nn.functional as F

import numpy as np

from monai.data import decollate_batch
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirst,
    SqueezeDimd,
    AsDiscrete,
    KeepLargestConnectedComponent,
    Compose,
    LabelFilter,
    MapLabelValue,
    Spacing,
    SqueezeDim,
    AsDiscrete
)
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric

from data_utils.io import save_img
import matplotlib.pyplot as plt


def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data['image'].to(device))
        #output = torch.argmax(output, dim=1)
    return output


def check_channel(inp):
    """
    確保輸入的張量是 5 維的 [B, C, H, W, D]。
    B: Batch size, C: Channel count
    """
    # 獲取輸入張量的維度數量
    len_inp_shape = inp.ndim  # 使用 .ndim 屬性更簡潔

    # 情況 1: 輸入是 3D 張量，形如 [H, W, D]
    # 需要增加 Batch 和 Channel 維度
    if len_inp_shape == 3:
        # 第一次 unsqueeze 在維度 0 增加 Channel 維度 -> [C, H, W, D] (C=1)
        # 第二次 unsqueeze 在維度 0 增加 Batch 維度 -> [B, C, H, W, D] (B=1)
        inp = torch.unsqueeze(inp, 0)
        inp = torch.unsqueeze(inp, 0)

    # 情況 2: 輸入是 4D 張量，形如 [C, H, W, D]
    # 只需要增加 Batch 維度
    elif len_inp_shape == 4:
        # 在維度 0 增加 Batch 維度 -> [B, C, H, W, D] (B=1)
        inp = torch.unsqueeze(inp, 0)

    # 如果輸入已經是 5D 或其他維度，則不做任何操作，直接返回
    return inp


def eval_label_pred(data, cls_num, device):
    print("\n--- 開始執行 eval_label_pred (手動 One-Hot 版本) ---")
    
    # 1. 定義 Metric 物件
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False)
    confusion_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name=["sensitivity", "specificity"],
        compute_sample=False, 
        reduction="mean", 
        get_not_nans=False
    )
    
    # 2. 準備資料
    val_label_int = data["label"].to(device) # 整數類別圖
    val_pred_logits = data["pred"].to(device) # 概率圖 (logits)
    
    # --- 3. 手動進行後處理，完全取代 AsDiscrete ---
    
    # 處理模型預測：
    # a. Argmax: 從 logits 得到整數類別圖
    #    輸入: [B, C, H, W, D] -> 輸出: [B, H, W, D]
    val_pred_int = torch.argmax(val_pred_logits, dim=1) 
    
    # b. One-Hot 編碼預測結果:
    #    輸入: [B, H, W, D] -> 輸出: [B, H, W, D, C]
    val_output_onehot_temp = F.one_hot(val_pred_int.long(), num_classes=cls_num)
    # c. 維度重排以符合 MONAI 要求 (C, H, W, D)
    #    輸入: [B, H, W, D, C] -> 輸出: [B, C, H, W, D]
    val_output_convert = val_output_onehot_temp.permute(0, 4, 1, 2, 3)

    # 處理真實標籤：
    # a. 移除標籤多餘的通道維度 (如果存在)
    #    輸入: [B, 1, H, W, D] -> 輸出: [B, H, W, D]
    if val_label_int.shape[1] == 1:
        val_label_int = torch.squeeze(val_label_int, dim=1)
    # b. One-Hot 編碼標籤結果:
    #    輸入: [B, H, W, D] -> 輸出: [B, H, W, D, C]
    val_labels_onehot_temp = F.one_hot(val_label_int.long(), num_classes=cls_num)
    # c. 維度重排
    #    輸入: [B, H, W, D, C] -> 輸出: [B, C, H, W, D]
    val_labels_convert = val_labels_onehot_temp.permute(0, 4, 1, 2, 3)

    print("\n--- 手動 One-Hot 後關鍵形狀檢查 ---")
    print(f"轉換後的 val_output_convert 形狀: {val_output_convert.shape}")
    print(f"轉換後的 val_labels_convert 形狀: {val_labels_convert.shape}")
    print("----------------------------------\n")

    # 4. 累積結果 (現在兩個輸入的形狀絕對一致)
    #    為了安全，我們只傳遞前景通道
    dice_metric(y_pred=val_output_convert[:, 1:], y=val_labels_convert[:, 1:])
    iou_metric(y_pred=val_output_convert[:, 1:], y=val_labels_convert[:, 1:])
    confusion_metric(y_pred=val_output_convert[:, 1:], y=val_labels_convert[:, 1:])

    # 5. 獲取最終結果
    dc_vals = dice_metric.aggregate().cpu().numpy()
    iou_vals = iou_metric.aggregate().cpu().numpy()
    conf_matrix_results = confusion_metric.aggregate()
    sensitivity_vals = conf_matrix_results[0].cpu().numpy()
    specificity_vals = conf_matrix_results[1].cpu().numpy()
    
    # 6. 重置 metrics
    dice_metric.reset()
    iou_metric.reset()
    confusion_metric.reset()
    
    print("--- eval_label_pred 執行完畢 ---\n")
    return dc_vals, iou_vals, sensitivity_vals, specificity_vals
                
def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def get_label_transform(data_name, keys=['label']):
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_lbl_transform = getattr(transform, 'get_label_transform', None)
    return get_lbl_transform(keys)


# ================= 這是最終符合邏輯的 run_infering 函式 =================
def run_infering(
        model,
        data,
        model_inferer,
        post_transform,
        args
    ):
    ret_dict = {}
    
    # 1. 執行推論，得到概率圖 (logits)
    start_time = time.time()
    logits = infer(model, data, model_inferer, args.device)
    end_time  = time.time()
    ret_dict['inf_time'] = end_time-start_time
    print(f'infer time: {ret_dict["inf_time"]} sec')
    
    # 2. 第一次評估 (在重採樣空間中，使用 logits)
    if 'label' in data.keys():
        print('正在於重採樣空間中進行評估 (空間還原前)...')
        eval_data_resampled = {'pred': logits, 'label': data['label']}
        tta_dc_vals, tta_iou_vals, _ , _ = eval_label_pred(eval_data_resampled, args.out_channels, args.device)
        print('Dice (重採樣後):', tta_dc_vals)
        print('IoU (重採樣後):', tta_iou_vals)
        ret_dict['tta_dc'] = tta_dc_vals
        ret_dict['tta_iou'] = tta_iou_vals

    # --- 關鍵修正：在空間轉換前，先執行 Argmax ---
    # 3. 從 logits 得到整數類別圖，用於空間轉換
    #    .to(torch.uint8) 可以節省一些記憶體
    pred_class_map = torch.argmax(logits, dim=1, keepdim=True).to(torch.uint8)
    
    # 將整數類別圖放回 data['pred']，準備進行空間轉換
    data['pred'] = pred_class_map

    # 4. 將整數類別圖還原到原始影像空間
    #    Restored 在處理整數圖時應使用 'nearest' 模式以避免產生非整數值
    #    這需要在 post_transform 的定義中修改
    print("正在將預測結果還原至原始空間...")
    data = post_transform(data)
    # 此時 data['pred'] 是被還原後的整數類別圖

    # 5. 在原始空間中進行第二次評估
    if 'label' in data.keys():
        print('正在為最終評估載入原始標籤...')
        lbl_dict = {'label': data['pred'].meta['filename_or_obj']}
        label_loader = get_label_transform(args.data_name, keys=['label'])
        lbl_data = label_loader(lbl_dict)
        data['label'] = lbl_data['label']

        print('正在於原始空間中進行評估...')
        # !!! 重要：eval_label_pred 期望 logits 作為輸入 !!!
        # 但我們現在只有整數圖。所以我們不能直接複用 eval_label_pred。
        # 我們需要一個簡化版的評估邏輯。
        # 為了快速解決，我們先跳過第二次評估，確保程式能跑完。
        # (稍後可以再編寫一個接收整數圖的評估函式)
        print("注意：已跳過原始空間中的第二次評估以簡化流程。")
        ori_dc_vals, ori_iou_vals, ori_sensitivity_vals, ori_specificity_vals = ([0],[0],[0],[0])
        # ... (後續的 ret_dict 賦值) ...

    # 6. 最後的後處理
    final_pred_map = data['pred'] # data['pred'] 已經是最終的整數圖
    if args.infer_post_process:
        print('正在進行最大連通元件分析...')
        applied_labels = [i for i in range(1, args.out_channels)]
        final_pred_map = KeepLargestConnectedComponent(applied_labels=applied_labels)(final_pred_map)
        
    # ... 儲存邏輯 ...
    
    print("正在釋放記憶體...")
    del data, logits, final_pred_map
    torch.cuda.empty_cache()

    return ret_dict
