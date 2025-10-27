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
        output = torch.argmax(output, dim=1)
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
    # --- 這是您需要修改的部分 ---
    # 1. 為真實標籤定義後處理 (保持不變)
    post_label = AsDiscrete(to_onehot=cls_num)
    
    # 2. 為模型預測定義一個新的、獨立的後處理
    #    關鍵在於 argmax=True
    post_pred = AsDiscrete(argmax=True, to_onehot=cls_num)
    # --- 修改部分結束 ---

    # metric definitions (保持不變)
    dice_metric = ...
    iou_metric = ...
    confusion_metric = ...
    
    # batch data (保持不變)
    val_label, val_pred = (data["label"].to(device), data["pred"].to(device))
    
    # check shape is 5 (保持不變)
    val_label = check_channel(val_label)
    val_pred = check_channel(val_pred)
    
    # --- 這是您需要修改的另一部分 ---
    # deallocate batch data
    # 對 label 使用 post_label
    val_labels_convert = [
        post_label(val_label_tensor) for val_label_tensor in val_label
    ]
    # 對 pred 使用新建的 post_pred
    val_output_convert = [
        post_pred(val_pred_tensor) for val_pred_tensor in val_pred
    ]
    # --- 修改部分結束 ---    
    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    iou_metric(y_pred=val_output_convert, y=val_labels_convert)
    confusion_metric(y_pred=val_output_convert, y=val_labels_convert)

    dc_vals = dice_metric.get_buffer().detach().cpu().numpy().squeeze()
    iou_vals = iou_metric.get_buffer().detach().cpu().numpy().squeeze()
    
    confusion_vals = confusion_metric.get_buffer().detach().cpu().numpy().squeeze()
    print("Confusion_Vals：", confusion_vals)
    tp = confusion_vals[:, 0]
    fp = confusion_vals[:, 1]
    tn = confusion_vals[:, 2]
    fn = confusion_vals[:, 3]
    sensitivity_vals = tp / (tp + fn)
    specificity_vals = tn / (tn + fp)
    
    
    return dc_vals, iou_vals, sensitivity_vals, specificity_vals


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
    
    
    # test
    start_time = time.time()
    data['pred'] = infer(model, data, model_inferer, args.device)
    end_time  = time.time()
    ret_dict['inf_time'] = end_time-start_time
    print(f'infer time: {ret_dict["inf_time"]} sec')
    
    # post process transform
    if args.infer_post_process:
        print('use post process infer')
        applied_labels = np.unique(data['pred'].flatten())[1:]
        data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(data['pred'])
    
    # eval infer tta
    if 'label' in data.keys():
        tta_dc_vals, tta_iou_vals, _ , _ = eval_label_pred(data, args.out_channels, args.device)
        print('infer test time aug:')
        print('dice:', tta_dc_vals)
        print('iou:', tta_iou_vals)
        ret_dict['tta_dc'] = tta_dc_vals
        ret_dict['tta_iou'] = tta_iou_vals
        
        # post label transform 
        sqz_transform = SqueezeDimd(keys=['label'])
        data = sqz_transform(data)
    
    # post transform
    data = post_transform(data)
    
    # eval infer origin
    if 'label' in data.keys():
        # get orginal label
        lbl_dict = {'label': data['image'].meta['filename_or_obj']}
        label_loader = get_label_transform(args.data_name, keys=['label'])
        lbl_data = label_loader(lbl_dict)
        
        data['label'] = lbl_data['label']
        data['label_meta_dict'] = lbl_data['label']
        
        ori_dc_vals, ori_iou_vals, ori_sensitivity_vals, ori_specificity_vals = eval_label_pred(data, args.out_channels, args.device)
        print('infer test original:')
        print('dice:', ori_dc_vals)
        print('iou:', ori_iou_vals)
        print('sensitivity:', ori_sensitivity_vals)
        print('specificity:', ori_specificity_vals)
        ret_dict['ori_dc'] = ori_dc_vals
        ret_dict['ori_iou'] = ori_iou_vals
        ret_dict['ori_sensitivity'] = ori_sensitivity_vals
        ret_dict['ori_specificity'] = ori_specificity_vals
    
    if args.data_name == 'mmwhs':
        mmwhs_transform = Compose([
            LabelFilter(applied_labels=[1, 2, 3, 4, 5, 6, 7]),
            MapLabelValue(orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                            target_labels=[0, 500, 600, 420, 550, 205, 820, 850]),
            # AddChannel(),
            # Spacing(
            #     pixdim=(args.space_x, args.space_y, args.space_z),
            #     mode=("nearest"),
            # ),
            # SqueezeDim()
        ])
        data['pred'] = mmwhs_transform(data['pred'])
        
    
    if not args.test_mode:
        # save pred result
        filename = get_filename(data)
        infer_img_pth = os.path.join(args.infer_dir, filename)

        save_img(
          data['pred'], 
          data['pred_meta_dict'], 
          infer_img_pth
        )
        
    return ret_dict
