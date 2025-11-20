import os
import time
import importlib
import gc
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
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric

from data_utils.io import save_img
import matplotlib.pyplot as plt


def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data['image'].to(device))
    return output

def eval_label_pred(data, cls_num, device):
    print("\n--- 開始執行 eval_label_pred ---")
    
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False) 
    confusion_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name=["sensitivity", "specificity"],
        compute_sample=False, 
        reduction="mean", 
        get_not_nans=False
    )
    
    # 使用 no_grad 節省記憶體
    with torch.no_grad():
        val_label_int = data["label"].to(device)
        val_pred_logits = data["pred"].to(device)
        
        val_pred_int = torch.argmax(val_pred_logits, dim=1) 
        
        # 轉 One-Hot
        val_output_onehot = F.one_hot(val_pred_int.long(), num_classes=cls_num).permute(0, 4, 1, 2, 3)
        
        if val_label_int.shape[1] == 1:
            val_label_int = torch.squeeze(val_label_int, dim=1)
        val_labels_onehot = F.one_hot(val_label_int.long(), num_classes=cls_num).permute(0, 4, 1, 2, 3)

        print(f"Shape check: Pred {val_output_onehot.shape}, Label {val_labels_onehot.shape}")

        # 累積結果
        dice_metric(y_pred=val_output_onehot[:, 1:], y=val_labels_onehot[:, 1:])
        iou_metric(y_pred=val_output_onehot[:, 1:], y=val_labels_onehot[:, 1:])
        confusion_metric(y_pred=val_output_onehot[:, 1:], y=val_labels_onehot[:, 1:])

        # 獲取結果
        dc_vals = dice_metric.aggregate().item()
        iou_vals = iou_metric.aggregate().item()
        conf_matrix_results = confusion_metric.aggregate()
        sensitivity_vals = conf_matrix_results[0].cpu().numpy()
        specificity_vals = conf_matrix_results[1].cpu().numpy()
    
    # 清理 metrics
    dice_metric.reset()
    iou_metric.reset()
    confusion_metric.reset()
    
    # 手動清理大張量
    del val_label_int, val_pred_logits, val_pred_int, val_output_onehot, val_labels_onehot
    torch.cuda.empty_cache()
    
    print("--- eval_label_pred 執行完畢 ---\n")
    return {
        "dice": dc_vals,
        "iou": iou_vals,
        "sensitivity": sensitivity_vals,
        "specificity": specificity_vals
    }

def eval_class_map(pred_map, label_map, cls_num, device):
    print("\n--- 開始執行 eval_class_map ---")
    
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False)
    confusion_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name=["sensitivity", "specificity"],
        compute_sample=False, 
        reduction="mean", 
        get_not_nans=False
    )

    with torch.no_grad():
        pred_map = torch.as_tensor(pred_map, device=device)
        label_map = torch.as_tensor(label_map, device=device)

        while pred_map.ndim < 3: pred_map = pred_map.unsqueeze(-1)
        while label_map.ndim < 3: label_map = label_map.unsqueeze(-1)

        target_spatial_size = label_map.shape[-3:] 
        
        if pred_map.shape[-3:] != target_spatial_size:
            print(f"Resize pred from {pred_map.shape[-3:]} to {target_spatial_size}")
            resize_transform = Resize(spatial_size=target_spatial_size, mode="nearest")
            pred_map = resize_transform(pred_map.unsqueeze(0)).squeeze(0)

        if pred_map.ndim == 3: pred_map = pred_map.unsqueeze(0)
        if label_map.ndim == 3: label_map = label_map.unsqueeze(0)
        
        pred_map = torch.clamp(pred_map, min=0, max=cls_num - 1)
        label_map = torch.clamp(label_map, min=0, max=cls_num - 1)
        
        # 轉 One-Hot
        pred_onehot = F.one_hot(pred_map.long(), num_classes=cls_num).permute(0, 4, 1, 2, 3)
        label_onehot = F.one_hot(label_map.long(), num_classes=cls_num).permute(0, 4, 1, 2, 3)
        
        # 累積結果
        dice_metric(y_pred=pred_onehot[:, 1:], y=label_onehot[:, 1:])
        iou_metric(y_pred=pred_onehot[:, 1:], y=label_onehot[:, 1:])
        confusion_metric(y_pred=pred_onehot[:, 1:], y=label_onehot[:, 1:])

        # 獲取結果
        dc_vals = dice_metric.aggregate().item()
        iou_vals = iou_metric.aggregate().item()
        conf_matrix_results = confusion_metric.aggregate()
        sensitivity_vals = conf_matrix_results[0].cpu().numpy()
        specificity_vals = conf_matrix_results[1].cpu().numpy()
    
    dice_metric.reset()
    iou_metric.reset()
    confusion_metric.reset()

    # 手動清理
    del pred_map, label_map, pred_onehot, label_onehot
    torch.cuda.empty_cache()
    
    print("--- eval_class_map 完畢 ---\n")
    
    return {
        "dice": dc_vals,
        "iou": iou_vals,
        "sensitivity": sensitivity_vals,
        "specificity": specificity_vals
    }
                    
def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def get_label_transform(data_name, keys=['label']):
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_lbl_transform = getattr(transform, 'get_label_transform', None)
    return get_lbl_transform(keys)


def run_infering(
        model,
        data,
        model_inferer,
        post_transform,
        args
    ):
    ret_dict = {}
    
    # 1. 備份 Meta 資訊
    original_meta = data['image'].meta.copy() 
    original_filename = original_meta['filename_or_obj']
    
    # 2. 推論 (產生 Logits)
    print("開始推論...")
    start_time = time.time()
    logits = infer(model, data, model_inferer, args.device)
    end_time  = time.time()
    ret_dict['inf_time'] = end_time-start_time
    print(f'Infer time: {ret_dict["inf_time"]} sec')
    
    # 3. 第一次評估 (使用 Logits)
    if 'label' in data.keys():
        print('重採樣空間評估...')
        eval_data = {'pred': logits, 'label': data['label']}
        metrics_result = eval_label_pred(eval_data, args.out_channels, args.device)
        
        # 評估完立刻刪除 eval_data 以釋放參考
        del eval_data
        gc.collect()

        ret_dict['tta_dc'] = metrics_result["dice"]
        ret_dict['tta_iou'] = metrics_result["iou"]
        ret_dict['tta_sensitivity'] = metrics_result["sensitivity"]
        ret_dict['tta_specificity'] = metrics_result["specificity"]
        print('Dice (重採樣後):', ret_dict['tta_dc']) 

    # 4. 生成整數類別圖 (並立刻刪除 Logits)
    print("轉換整數圖並釋放 Logits...")
    pred_class_map = torch.argmax(logits, dim=1, keepdim=False).to(torch.uint8)
    
    # --- 關鍵修正：立刻刪除最大的變數 logits ---
    del logits 
    torch.cuda.empty_cache()
    gc.collect()
    # ---------------------------------------

    data['pred'] = pred_class_map
    data['image'] = data['image'].meta 

    # 5. 還原到原始空間
    print("還原至原始空間...")
    data = post_transform(data)
    # 這裡 data['pred'] 變成 numpy array，pred_class_map (GPU tensor) 可以刪了
    del pred_class_map
    gc.collect()
    
    # 6. 第二次評估 (在原始空間)
    if 'label' in data.keys():
        print('載入原始標籤進行評估...')
        lbl_dict = {'label': original_filename}
        label_loader = get_label_transform(args.data_name, keys=['label'])
        lbl_data = label_loader(lbl_dict)
        
        # 呼叫評估
        metrics_result_ori = eval_class_map(
            pred_map=data['pred'], 
            label_map=lbl_data['label'], 
            cls_num=args.out_channels, 
            device=args.device
        )
        
        # 評估完立刻刪除標籤資料
        del lbl_data
        gc.collect()
        
        ret_dict['ori_dc'] = metrics_result_ori["dice"]
        ret_dict['ori_iou'] = metrics_result_ori["iou"]
        ret_dict['ori_sensitivity'] = metrics_result_ori["sensitivity"]
        ret_dict['ori_specificity'] = metrics_result_ori["specificity"]
        
        print('Dice (原始空間):', ret_dict['ori_dc']) 

    # 7. 儲存結果
    if args.test_mode:
        print("儲存預測圖...")
        filename = PurePath(original_filename).name
        infer_img_pth = os.path.join(args.eval_dir, filename)
        save_img(
          data['pred'], 
          original_meta, 
          infer_img_pth
        )
        print(f"Saved: {infer_img_pth}")
        
    print("本輪結束，最終清理...")
    del data
    torch.cuda.empty_cache()
    gc.collect()

    return ret_dict
