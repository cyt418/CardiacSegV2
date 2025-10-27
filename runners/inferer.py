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
    AsDiscrete,
    Resize
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
    return {
        "dice": dc_vals,
        "iou": iou_vals,
        "sensitivity": sensitivity_vals,
        "specificity": specificity_vals
    }

# ================= 這是 eval_class_map 的最終完整正確版 =================
def eval_class_map(pred_map, label_map, cls_num, device):
    print("\n--- 開始執行 eval_class_map (處理整數圖) ---")
    
    # 確保兩者都是 PyTorch 張量並在正確的設備上
    pred_map = torch.as_tensor(pred_map, device=device)
    label_map = torch.as_tensor(label_map, device=device)

    # 確保兩者至少是 3D 空間張量 (H, W, D)
    while pred_map.ndim < 3: pred_map = pred_map.unsqueeze(-1)
    while label_map.ndim < 3: label_map = label_map.unsqueeze(-1)

    # --- 強制統一空間尺寸 ---
    target_spatial_size = label_map.shape[-3:] # 取最後三個維度 (H, W, D)
    
    if pred_map.shape[-3:] != target_spatial_size:
        print(f"尺寸不匹配！正在將 pred 從 {pred_map.shape[-3:]} resize 到 {target_spatial_size}")
        resize_transform = Resize(spatial_size=target_spatial_size, mode="nearest")
        # Resize 期望輸入是 [C, H, W, D]，所以我們先 unsqueeze
        pred_map = resize_transform(pred_map.unsqueeze(0)).squeeze(0)

    # --- ！！！補上被我遺漏的 Metric 物件定義！！！ ---
    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False)
    # ---------------------------------------------------

    # --- 手動進行 One-Hot 編碼 ---
    # 確保輸入是 4D 的批次格式 [B, H, W, D] 以進行 one-hot
    if pred_map.ndim == 3: pred_map = pred_map.unsqueeze(0)
    if label_map.ndim == 3: label_map = label_map.unsqueeze(0)
    
    # --- 關鍵修正：在 one-hot 編碼前，強制裁剪數值範圍 ---
    # torch.clamp 會將所有小於 0 的值設為 0，大於 cls_num-1 的值設為 cls_num-1
    pred_map = torch.clamp(pred_map, min=0, max=cls_num - 1)
    label_map = torch.clamp(label_map, min=0, max=cls_num - 1)
    # ----------------------------------------------------
    
    pred_map = torch.clamp(pred_map, min=0, max=cls_num - 1)
    label_map = torch.clamp(label_map, min=0, max=cls_num - 1)
    
    pred_onehot = F.one_hot(pred_map.long(), num_classes=cls_num).permute(0, 4, 1, 2, 3)
    label_onehot = F.one_hot(label_map.long(), num_classes=cls_num).permute(0, 4, 1, 2, 3)
    
    # --- 累積結果 ---
    dice_metric(y_pred=pred_onehot[:, 1:], y=label_onehot[:, 1:])
    iou_metric(y_pred=pred_onehot[:, 1:], y=label_onehot[:, 1:])

    # --- 獲取結果 ---
    dc_vals = dice_metric.aggregate().cpu().numpy()
    iou_vals = iou_metric.aggregate().cpu().numpy()
    
    # --- 重置 ---
    dice_metric.reset()
    iou_metric.reset()
    
    print("--- eval_class_map 完畢 ---\n")
    return dc_vals, iou_vals
                    
def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def get_label_transform(data_name, keys=['label']):
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_lbl_transform = getattr(transform, 'get_label_transform', None)
    return get_lbl_transform(keys)


# ================= 這是 run_infering 的最終完整版 =================
def run_infering(
        model,
        data,
        model_inferer,
        post_transform,
        args
    ):
    ret_dict = {}
    original_meta = data['image'].meta.copy() 
    original_filename = data['image'].meta['filename_or_obj']
    
    # 1. 推論得到 logits
    start_time = time.time()
    logits = infer(model, data, model_inferer, args.device)
    end_time  = time.time()
    ret_dict['inf_time'] = end_time-start_time
    print(f'infer time: {ret_dict["inf_time"]} sec')
    
    # 2. 第一次評估 (在重採樣空間，使用 logits)
    if 'label' in data.keys():
        print('正在於重採樣空間中進行評估...')
        eval_data = {'pred': logits, 'label': data['label']}
        # 呼叫處理 logits 的舊版評估函式
        metrics_result = eval_label_pred(eval_data, args.out_channels, args.device)
        ret_dict['tta_dc'] = metrics_result["dice"]
        ret_dict['tta_iou'] = metrics_result["iou"]
        # 為了讓 DataFrame 格式一致，我們也填充 sensitivity 和 specificity
        ret_dict['tta_sensitivity'] = metrics_result["sensitivity"]
        ret_dict['tta_specificity'] = metrics_result["specificity"]
        print('Dice (重採樣後):', ret_dict['tta_dc'])

    # 3. 從 logits 得到整數類別圖，用於空間轉換
    pred_class_map = torch.argmax(logits, dim=1, keepdim=False).to(torch.uint8)
    data['pred'] = pred_class_map
    data['image'] = data['image'].meta # 確保 Restored 能找到參考影像的 meta

    # 4. 還原到原始空間
    print("正在將預測結果還原至原始空間...")
    data = post_transform(data)
    # 此時 data['pred'] 是被還原後的整數類別圖 (NumPy 陣列)
    
    # 5. 第二次評估 (在原始空間)
    if 'label' in data.keys():
        print('正在為最終評估載入原始標籤...')
        lbl_dict = {'label': original_filename}
        label_loader = get_label_transform(args.data_name, keys=['label'])
        lbl_data = label_loader(lbl_dict)
        
        print('正在於原始空間中進行評估...')
        # 呼叫我們新建的、處理整數圖的評估函式
        # data['pred'] 是 NumPy 陣列, lbl_data['label'] 是 PyTorch 張量，eval_class_map 都能處理
        ori_dc_vals, ori_iou_vals = eval_class_map(
            pred_map=data['pred'], 
            label_map=lbl_data['label'], 
            cls_num=args.out_channels, 
            device=args.device
        )
        ret_dict['ori_dc'] = ori_dc_vals
        ret_dict['ori_iou'] = ori_iou_vals
        # 暫時用 0 填充 sensitivity 和 specificity
        num_fg_classes = args.out_channels - 1
        ret_dict['ori_sensitivity'] = np.zeros(num_fg_classes)
        ret_dict['ori_specificity'] = np.zeros(num_fg_classes)
        print('Dice (原始空間):', ret_dict['ori_dc'])

    # 6. 儲存最終的預測結果
    if args.test_mode:
        print("正在儲存預測結果...")
        filename = PurePath(original_filename).name
        infer_img_pth = os.path.join(args.eval_dir, filename)
        save_img(data['pred'], original_meta, infer_img_pth)
        print(f"結果已儲存至: {infer_img_pth}")
        
    print("正在釋放記憶體...")
    del data, logits, pred_class_map
    torch.cuda.empty_cache()

    return ret_dict
