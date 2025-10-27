import os
import time
import importlib
from pathlib import PurePath

import torch

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
    # --- 1. 定義後處理轉換 ---
    # 為模型預測定義轉換 (包含 argmax)
    post_pred = AsDiscrete(argmax=True, to_onehot=cls_num)
    # 為真實標籤定義轉換
    post_label = AsDiscrete(to_onehot=cls_num)
    
    # --- 2. 正確地定義所有 Metric 物件 ---
    # 確保這些定義沒有被意外覆蓋
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )
    iou_metric = MeanIoU(include_background=False)
    confusion_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name=["sensitivity", "specificity"], # 一次性計算多個指標
        compute_sample=False, 
        reduction="mean", 
        get_not_nans=False
    )
    
    # --- 3. 準備資料 ---
    val_label, val_pred = (data["label"].to(device), data["pred"].to(device))
    val_label = check_channel(val_label)
    val_pred = check_channel(val_pred)
    
    # --- 4. 應用後處理轉換 ---
    # MONAI 的轉換可以直接處理整個批次 (batch)
    val_output_convert = post_pred(val_pred)
    val_labels_convert = post_label(val_label)
    
    # --- 5. 累積結果 ---
    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    iou_metric(y_pred=val_output_convert, y=val_labels_convert)
    confusion_metric(y_pred=val_output_convert, y=val_labels_convert)

    # --- 6. 使用 .aggregate() 獲取最終結果 ---
    dc_vals = dice_metric.aggregate().cpu().numpy()
    iou_vals = iou_metric.aggregate().cpu().numpy()
    conf_matrix_results = confusion_metric.aggregate()
    sensitivity_vals = conf_matrix_results[0].cpu().numpy() # sensitivity 是第一個結果
    specificity_vals = conf_matrix_results[1].cpu().numpy() # specificity 是第二個結果
    
    # --- 7. 重置 metrics 以備下次呼叫 ---
    dice_metric.reset()
    iou_metric.reset()
    confusion_metric.reset()
    
    return dc_vals, iou_vals, sensitivity_vals, specificity_vals

def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def get_label_transform(data_name, keys=['label']):
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_lbl_transform = getattr(transform, 'get_label_transform', None)
    return get_lbl_transform(keys)


# ================= 這是新的 run_infering 函式 =================
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
    
    # 將 logits 放入 data 字典，用於第一次評估和後續的 Restore
    data['pred'] = logits

    # 2. 在重採樣空間中進行第一次評估 (之前被標記為 TTA 的部分)
    if 'label' in data.keys():
        print('正在於重採樣空間中進行評估 (空間還原前)...')
        # eval_label_pred 期望 logits 作為 pred
        tta_dc_vals, tta_iou_vals, _ , _ = eval_label_pred(data, args.out_channels, args.device)
        print('Dice (重採樣後):', tta_dc_vals)
        print('IoU (重採樣後):', tta_iou_vals)
        ret_dict['tta_dc'] = tta_dc_vals
        ret_dict['tta_iou'] = tta_iou_vals
    
    # 3. 將概率圖還原到原始影像空間
    # post_transform (包含 Restored) 應該作用在概率圖上
    print("正在將預測結果還原至原始空間...")
    data = post_transform(data)
    # 此時 data['pred'] 是被還原後的概率圖

    # 4. 在原始空間中進行第二次評估
    if 'label' in data.keys():
        # 載入原始空間的標籤
        print('正在為最終評估載入原始標籤...')
        # 從還原後的 meta data 中獲取原始檔名
        lbl_dict = {'label': data['pred'].meta['filename_or_obj']} 
        label_loader = get_label_transform(args.data_name, keys=['label'])
        lbl_data = label_loader(lbl_dict)
        data['label'] = lbl_data['label']

        print('正在於原始空間中進行評估...')
        # eval_label_pred 期望 logits 作為 pred，而 data['pred'] 正是還原後的 logits
        ori_dc_vals, ori_iou_vals, ori_sensitivity_vals, ori_specificity_vals = eval_label_pred(data, args.out_channels, args.device)
        print('Dice (原始空間):', ori_dc_vals)
        print('IoU (原始空間):', ori_iou_vals)
        print('Sensitivity (原始空間):', ori_sensitivity_vals)
        print('Specificity (原始空間):', ori_specificity_vals)
        ret_dict['ori_dc'] = ori_dc_vals
        ret_dict['ori_iou'] = ori_iou_vals
        ret_dict['ori_sensitivity'] = ori_sensitivity_vals
        ret_dict['ori_specificity'] = ori_specificity_vals

    # 5. 最後，在所有評估完成後，才進行最終的後處理以準備儲存
    print('正在為儲存檔案做最後處理...')
    
    # 從還原後的 logits 得到最終的整數類別圖
    final_pred_map = torch.argmax(data['pred'], dim=1, keepdim=True)

    if args.infer_post_process:
        print('正在進行最大連通元件分析...')
        # KeepLargestConnectedComponent 需要整數輸入，我們剛剛已經得到了
        applied_labels = [i for i in range(1, args.out_channels)] # 例如: [1, 2, 3]
        final_pred_map = KeepLargestConnectedComponent(applied_labels=applied_labels)(final_pred_map)
    
    # ... 您儲存最終結果的邏輯 ...
    # if not args.test_mode:
    #     filename = get_filename(data)
    #     infer_img_pth = os.path.join(args.infer_dir, filename)
    #     save_img(final_pred_map, data['pred'].meta, infer_img_pth)
        
    # --- 記憶體管理：在函式結束前釋放大型張量 ---
    print("正在釋放記憶體...")
    del data
    del logits
    del final_pred_map
    torch.cuda.empty_cache()
    # ------------------------------------

    return ret_dict
