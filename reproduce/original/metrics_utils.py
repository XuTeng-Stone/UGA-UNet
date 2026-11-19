import numpy as np
import torch
import copy # [新增] 引入 copy 模块
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, generate_binary_structure
try:
    from thop import profile
except ImportError:
    profile = None

# ================= FLOPs & Params (修复版) =================
def cal_params_flops(model, input_size=(1, 3, 256, 256), device='cuda'):
    if profile is None:
        return 0.0, 0.0
    
    try:
        # [关键修复] 使用 deepcopy 复制模型
        # 这样 thop 的 hooks 只会挂载在副本上，不会污染主模型
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(device)
        model_copy.eval()
        
        input_tensor = torch.randn(input_size).to(device)
        
        # 计算 FLOPs
        flops, params = profile(model_copy, inputs=(input_tensor, ), verbose=False)
        
        # 转换为 GFLOPs 和 MParams
        flops_g = flops / 1e9
        params_m = params / 1e6
        
        # 手动删除副本，确保释放显存
        del model_copy, input_tensor
        torch.cuda.empty_cache()
        
        return params_m, flops_g

    except Exception as e:
        print(f"[Warning] FLOPs calculation failed: {e}")
        # 如果计算失败，返回 0，但不中断程序
        return 0.0, 0.0

# ================= HD95 & ASSD (保持不变) =================
def get_surface_distance(mask_gt, mask_pred, spacing_mm=1.0):
    if mask_gt.sum() == 0 or mask_pred.sum() == 0:
        return np.nan, np.nan

    def get_border_pixels(mask):
        struct = generate_binary_structure(2, 1)
        eroded = distance_transform_edt(mask) > 1 
        border = np.logical_xor(mask, eroded)
        return np.argwhere(border)

    gt_border = get_border_pixels(mask_gt)
    pred_border = get_border_pixels(mask_pred)

    if len(gt_border) == 0 or len(pred_border) == 0:
        return np.nan, np.nan

    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_border)
    tree_pred = cKDTree(pred_border)
    
    d_p2g, _ = tree_gt.query(pred_border) 
    d_g2p, _ = tree_pred.query(gt_border) 
    
    assd = (d_p2g.sum() + d_g2p.sum()) / (len(d_p2g) + len(d_g2p))
    hd95 = np.percentile(np.hstack([d_p2g, d_g2p]), 95)
    
    return assd, hd95

def calculate_advanced_metrics(logits, target):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets = target.cpu().numpy()
    
    hd95_list = []
    assd_list = []
    
    for i in range(preds.shape[0]):
        p = preds[i]
        t = targets[i]
        if t.sum() > 0 and p.sum() > 0:
            assd, hd95 = get_surface_distance(t, p)
            if not np.isnan(assd):
                assd_list.append(assd)
                hd95_list.append(hd95)
    
    return np.mean(assd_list) if assd_list else 0.0, np.mean(hd95_list) if hd95_list else 0.0