import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
try:
    from thop import profile
except ImportError:
    profile = None

# ================= FLOPs & Params =================
def cal_params_flops(model, input_size=(1, 3, 256, 256), device='cuda'):
    if profile is None:
        return 0.0, 0.0
    
    model_eval = model.to(device)
    model_eval.eval()
    input_tensor = torch.randn(input_size).to(device)
    
    flops, params = profile(model_eval, inputs=(input_tensor, ), verbose=False)
    # 转换为 GFLOPs 和 MParams
    flops_g = flops / 1e9
    params_m = params / 1e6
    return params_m, flops_g

# ================= HD95 & ASSD (NumPy Version) =================
def get_surface_distance(mask_gt, mask_pred, spacing_mm=1.0):
    """
    计算表面距离，用于 ASSD 和 HD95
    mask_gt, mask_pred: shape (H, W), values 0 or 1
    """
    # 提取边界
    # 简单的边界提取：erode 前景，然后用原图减去
    # 或者用 distance_transform_edt 找 0 等值面
    
    if mask_gt.sum() == 0 or mask_pred.sum() == 0:
        return np.nan, np.nan # 无法计算

    # 计算距离变换
    # edt 计算的是到最近背景(0)点的距离。
    # 我们需要的是：对于 Pred 的边界点，找到 GT 边界的最近距离
    
    # 这种方法对于纯 Python 实现比较慢，我们用简单的边界点集合计算
    def get_border_pixels(mask):
        struct = generate_binary_structure(2, 1)
        # 腐蚀
        eroded = distance_transform_edt(mask) > 1 
        border = np.logical_xor(mask, eroded)
        return np.argwhere(border) # (N, 2) 坐标

    gt_border = get_border_pixels(mask_gt)
    pred_border = get_border_pixels(mask_pred)

    if len(gt_border) == 0 or len(pred_border) == 0:
        return np.nan, np.nan

    # 计算 Pred 边界点到 GT 边界的距离 (Directed Hausdorff A->B)
    d_pred_to_gt = []
    # 使用 scipy 的 cdist 太耗内存，使用 directed_hausdorff 只能算 max
    # 我们需要所有点的距离来算 ASSD
    
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_border)
    tree_pred = cKDTree(pred_border)
    
    d_p2g, _ = tree_gt.query(pred_border) # Pred 点到最近 GT 点的距离
    d_g2p, _ = tree_pred.query(gt_border) # GT 点到最近 Pred 点的距离
    
    # ASSD: Average Symmetric Surface Distance
    assd = (d_p2g.sum() + d_g2p.sum()) / (len(d_p2g) + len(d_g2p))
    
    # HD95: 95th percentile of distances
    hd95 = np.percentile(np.hstack([d_p2g, d_g2p]), 95)
    
    return assd, hd95

def calculate_advanced_metrics(logits, target):
    """
    logits: (B, C, H, W)
    target: (B, H, W)
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy() # (B, H, W)
    targets = target.cpu().numpy()
    
    hd95_list = []
    assd_list = []
    
    for i in range(preds.shape[0]):
        p = preds[i]
        t = targets[i]
        
        # 只在有肿瘤时计算
        if t.sum() > 0 and p.sum() > 0:
            assd, hd95 = get_surface_distance(t, p)
            if not np.isnan(assd):
                assd_list.append(assd)
                hd95_list.append(hd95)
        # 如果 GT 有肿瘤但 Pred 没有，可以设置一个惩罚值或者忽略，通常忽略或设最大图径
    
    return np.mean(assd_list) if assd_list else 0.0, np.mean(hd95_list) if hd95_list else 0.0