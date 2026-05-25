# 3D Object Detection for Autonomous Vehicles - Improved Hint (v2)

> **Why did best_solution01.py score only 0.0169?**
> Root-cause analysis of the previous attempt and concrete fixes are embedded throughout this hint.

---

## ROOT CAUSE ANALYSIS (Read This First!)

The previous solution scored 0.0169 (bronze = 0.046) due to six compounding mistakes:

| # | Bug | Impact |
|---|-----|--------|
| 1 | **Fake PointPillars**: stacks BEV height-slices instead of learning pillar features | Very High |
| 2 | **Missing sensor→ego calibration**: `transform_points_to_ego` is a no-op; points stay in sensor frame | High |
| 3 | **No TTA at all**: single forward pass despite MANDATORY requirement | High |
| 4 | **Max-pool only NMS**: no rotated-box IoU NMS → duplicated detections | High |
| 5 | **Output stride = 4**: 160×160 output at 0.64 m/pixel — pedestrian (0.7 m wide) = 1 pixel | Medium |
| 6 | **No class anchors**: regressing absolute log(w/l/h) instead of anchor residuals is very hard to converge | Medium |

**Fix all six and the score should comfortably reach bronze (0.046+).**

---

## 1. Set Clear Natural Language Instructions

You are an MLEBench agent competing in the Lyft 3D Object Detection competition.
Your goal: write a **Kaggle Medal-Winning (Top 10%, score ≥ 0.053 silver)** `runfile_0.py`.

**Hard Requirements (violation = guaranteed failure):**
- **BAN fake BEV projection**: Do NOT stack BEV height-slice channels via `points_to_bev`. That is NOT PointPillars.
- **IMPLEMENT real PointPillars**: use `PillarFeatureNet` (mini-PointNet per pillar) + 2D backbone. Keep voxelization scatter ops on CPU inside DataLoader workers.
- **APPLY sensor→ego calibration**: read `calibrated_sensor.json`; apply the sensor-to-ego rotation+translation before any box assignment. Without this, all point positions are wrong.
- **IMPLEMENT TTA** (flip_x, flip_y, rot_180) and merge via Weighted-Box-Fusion (WBF) or soft-NMS at inference.
- **IMPLEMENT rotated-NMS**: after decoding, filter with a BEV IoU NMS (use shapely or manual polygon intersection) at threshold 0.2.
- **REDUCE output stride to 2** (not 4): backbone+neck should output (BEV_H/2, BEV_W/2) so pedestrians occupy ≥ 2 output pixels.
- **USE class-specific anchor sizes** for residual regression: predict `(dx, dy, log(w/wa), log(l/la), sin(yaw), cos(yaw), dz, log(h/ha))` where (wa, la, ha) are the class anchor.
- Train ≥ 60 epochs with OneCycleLR. Batch size 16 on H100 80GB.
- Mandatory GT-Augmentation (Database Sampling).
- Correct World↔Ego coordinate conversions at all stages.
- Heatmap bias init with prior pi=0.01.

---

## 2. Data Loading & Sensor Calibration (CRITICAL FIX)

### 2.1 Build Calibrated Sample Index

```python
# SCoT – build_sample_index (Fixed)
# Sequential:
#   step 1 → load sample_data.json, ego_pose.json, calibrated_sensor.json
#   step 2 → for each LiDAR sample_data entry, extract:
#            ego_pose, sensor_calibration (sensor→ego rotation + translation)
#   step 3 → store token → {lidar_path, ego_pose, sensor2ego_rot, sensor2ego_trans}
# Branch:
#   if calibrated_sensor_token missing → fall back to identity transform (log warning)
# Output:
#   token_to_info: dict[str, dict]

import json, os, math
import numpy as np

def quat_to_rot(q):
    """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)

def build_sample_index(data_dir, lidar_dir):
    """
    Returns token_to_info: {sample_token: {lidar_path, ego_pose, s2e_R, s2e_t}}
    where s2e_R, s2e_t transform points from sensor frame → ego frame.
    """
    with open(os.path.join(data_dir, "sample_data.json")) as f:
        sample_data = json.load(f)
    with open(os.path.join(data_dir, "ego_pose.json")) as f:
        ego_poses = {ep["token"]: ep for ep in json.load(f)}
    # Load calibrated_sensor if exists
    cal_sensor_path = os.path.join(data_dir, "calibrated_sensor.json")
    cal_sensors = {}
    if os.path.exists(cal_sensor_path):
        with open(cal_sensor_path) as f:
            for cs in json.load(f):
                cal_sensors[cs["token"]] = cs

    token_to_info = {}
    for sd in sample_data:
        filename = sd.get("filename", "")
        if "lidar" not in filename.lower():
            continue
        basename = os.path.basename(filename)
        lidar_path = os.path.join(lidar_dir, basename)
        if not os.path.exists(lidar_path):
            continue

        sample_token = sd["sample_token"]
        ep = ego_poses.get(sd.get("ego_pose_token", ""), {})

        # Sensor → ego calibration
        cs_token = sd.get("calibrated_sensor_token", "")
        cs = cal_sensors.get(cs_token, {})
        if cs:
            s2e_R = quat_to_rot(cs["rotation"])   # sensor frame → ego frame
            s2e_t = np.array(cs["translation"])
        else:
            s2e_R = np.eye(3)
            s2e_t = np.zeros(3)

        token_to_info[sample_token] = {
            "lidar_path": lidar_path,
            "ego_pose": ep,
            "s2e_R": s2e_R,
            "s2e_t": s2e_t,
        }
    return token_to_info


def load_lidar_in_ego_frame(lidar_path, s2e_R, s2e_t):
    """Load LiDAR binary and transform points from sensor frame to ego frame."""
    raw = np.fromfile(lidar_path, dtype=np.float32)
    for ncols in [5, 4, 3]:
        if len(raw) % ncols == 0:
            pts = raw.reshape(-1, ncols)[:, :3]
            break
    else:
        return np.zeros((1, 3), dtype=np.float32)
    # Apply sensor → ego transform (CRITICAL FIX)
    pts_ego = (s2e_R @ pts.T).T + s2e_t   # (N,3) in ego frame
    return pts_ego.astype(np.float32)
```

---

## 3. Real PointPillars Architecture (CRITICAL FIX)

**DO NOT** stack height-slice channels. Implement real PointPillars:

```python
# SCoT – PointPillars voxelization
# Sequential:
#   step 1 → quantize each point to (pillar_x, pillar_y) index
#   step 2 → for each non-empty pillar, collect up to MAX_POINTS_PER_PILLAR points
#   step 3 → augment each point with (xc, yc, xp, yp) offsets to pillar/cluster centre
#   step 4 → encode with PillarFeatureNet (shared MLP → max-pool per pillar)
#   step 5 → scatter pillar features back to BEV pseudo-image (H_bev, W_bev)
#   step 6 → pass pseudo-image through 2D backbone + FPN neck
# Branch:
#   if pillar has 0 points → skip (already 0 in scattered output)
# Loop:
#   over non-empty pillar indices (vectorised via scatter_add)
# Constraints:
#   keep voxelization on CPU inside DataLoader workers; only scatter to GPU in model.forward

import torch, torch.nn as nn, torch.nn.functional as F

# ---- Config ----
class Config:
    x_min, x_max = -50.0, 50.0
    y_min, y_max = -50.0, 50.0
    z_min, z_max = -3.0,  5.0
    voxel_size   = 0.16          # metres per pillar (fine resolution)
    bev_w = int((x_max - x_min) / voxel_size)   # 625 → use 624 for divisibility
    bev_h = int((y_max - y_min) / voxel_size)
    MAX_PILLARS   = 20000
    MAX_PTS_PILLAR = 32
    pillar_feat_dim = 64         # PillarFeatureNet output channels
    # NOTE: output stride = 2 (not 4!) for pedestrian detection
    out_stride = 2
    out_w = bev_w // out_stride
    out_h = bev_h // out_stride

    classes = ["car","motorcycle","bus","bicycle","truck",
               "pedestrian","other_vehicle","animal","emergency_vehicle"]
    num_classes = len(classes)
    class2idx   = {c: i for i, c in enumerate(classes)}
    idx2class   = {i: c for i, c in enumerate(classes)}

    # Class-specific anchor sizes (w, l, h) in ego metres
    anchor_sizes = {
        "car":               (1.93, 4.63, 1.72),
        "motorcycle":        (0.77, 2.10, 1.48),
        "bus":               (2.92,12.01, 3.44),
        "bicycle":           (0.60, 1.76, 1.38),
        "truck":             (2.52, 6.93, 2.84),
        "pedestrian":        (0.72, 0.73, 1.77),
        "other_vehicle":     (1.98, 5.12, 2.01),
        "animal":            (0.51, 0.99, 0.82),
        "emergency_vehicle": (2.18, 5.48, 2.21),
    }
    batch_size  = 16
    num_epochs  = 60
    lr          = 2e-3
    weight_decay= 1e-4
    num_workers = 8
    input_dir   = "./input"
    output_dir  = "./submission"
    working_dir = "./working"

cfg = Config()


def voxelize(points, cfg):
    """
    CPU voxelization for DataLoader workers.
    Input:  points (N,3) float32 — ego frame
    Output: pillars    (P, MAX_PTS, 9) float32   (augmented point features)
            pillar_idx (P,)             int64     (flattened BEV index)
            num_pts    (P,)             int32
    """
    x_res = (cfg.x_max - cfg.x_min) / cfg.bev_w
    y_res = (cfg.y_max - cfg.y_min) / cfg.bev_h

    # Filter to BEV range
    mask = ((points[:, 0] >= cfg.x_min) & (points[:, 0] < cfg.x_max) &
            (points[:, 1] >= cfg.y_min) & (points[:, 1] < cfg.y_max) &
            (points[:, 2] >= cfg.z_min) & (points[:, 2] < cfg.z_max))
    pts = points[mask]

    if len(pts) == 0:
        pillars   = np.zeros((1, cfg.MAX_PTS_PILLAR, 9), dtype=np.float32)
        pidx      = np.zeros((1,), dtype=np.int64)
        npts      = np.zeros((1,), dtype=np.int32)
        return pillars, pidx, npts

    xi = np.floor((pts[:, 0] - cfg.x_min) / x_res).astype(np.int32)
    yi = np.floor((pts[:, 1] - cfg.y_min) / y_res).astype(np.int32)
    xi = np.clip(xi, 0, cfg.bev_w - 1)
    yi = np.clip(yi, 0, cfg.bev_h - 1)
    flat = yi * cfg.bev_w + xi

    # Group by pillar
    order = np.argsort(flat, kind='stable')
    flat_s, pts_s = flat[order], pts[order]

    uniq_flat, inv, counts = np.unique(flat_s, return_inverse=True, return_counts=True)
    if len(uniq_flat) > cfg.MAX_PILLARS:
        # Keep pillars with most points first
        keep = np.argsort(-counts)[:cfg.MAX_PILLARS]
        valid = np.isin(inv, keep)
        flat_s, pts_s, inv, uniq_flat = flat_s[valid], pts_s[valid], inv[valid], uniq_flat
        uniq_flat, inv, counts = np.unique(flat_s, return_inverse=True, return_counts=True)

    P = len(uniq_flat)
    pillars = np.zeros((P, cfg.MAX_PTS_PILLAR, 9), dtype=np.float32)
    npts    = np.zeros(P, dtype=np.int32)

    # Pillar centres in ego frame
    pi_x = (uniq_flat % cfg.bev_w + 0.5) * x_res + cfg.x_min
    pi_y = (uniq_flat // cfg.bev_w + 0.5) * y_res + cfg.y_min

    for p_i, p_flat in enumerate(uniq_flat):
        sel = pts_s[inv == p_i]
        n   = min(len(sel), cfg.MAX_PTS_PILLAR)
        npts[p_i] = n
        sel = sel[:n]
        # Point features: x, y, z, x-xc, y-yc, z-zc_mean, x-xp, y-yp, (pad z offset)
        xc, yc = sel[:, 0].mean(), sel[:, 1].mean()
        zc = sel[:, 2].mean()
        pillars[p_i, :n, 0] = sel[:, 0]
        pillars[p_i, :n, 1] = sel[:, 1]
        pillars[p_i, :n, 2] = sel[:, 2]
        pillars[p_i, :n, 3] = sel[:, 0] - xc        # offset to cluster centre
        pillars[p_i, :n, 4] = sel[:, 1] - yc
        pillars[p_i, :n, 5] = sel[:, 2] - zc
        pillars[p_i, :n, 6] = sel[:, 0] - pi_x[p_i] # offset to pillar centre
        pillars[p_i, :n, 7] = sel[:, 1] - pi_y[p_i]
        pillars[p_i, :n, 8] = (sel[:, 2] - cfg.z_min) / (cfg.z_max - cfg.z_min)

    return pillars, uniq_flat.astype(np.int64), npts


class PillarFeatureNet(nn.Module):
    """Shared MLP + max-pool over points within each pillar."""
    def __init__(self, in_dim=9, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, out_dim, bias=False), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, pillars, num_pts):
        """
        pillars:  (P, M, 9)
        num_pts:  (P,)
        returns:  (P, out_dim)
        """
        P, M, C = pillars.shape
        pts_flat = pillars.view(P * M, C)
        feats    = self.mlp(pts_flat).view(P, M, self.out_dim)
        # Mask padding
        mask = torch.arange(M, device=pillars.device).unsqueeze(0) < num_pts.unsqueeze(1)
        feats = feats * mask.unsqueeze(-1).float()
        return feats.max(dim=1).values   # (P, out_dim)


def scatter_to_bev(pillar_feats, pillar_idx, bev_h, bev_w):
    """Scatter (P, C) pillar features to (C, H, W) pseudo-image."""
    C = pillar_feats.shape[1]
    bev = torch.zeros(C, bev_h * bev_w, device=pillar_feats.device)
    idx = pillar_idx.unsqueeze(0).expand(C, -1)   # (C, P)
    bev.scatter_(1, idx, pillar_feats.T)
    return bev.view(C, bev_h, bev_w)


class BEVBackbone(nn.Module):
    """Lightweight 2D backbone for PointPillars pseudo-image.
       Output stride = 2 (not 4) for better small-object resolution.
    """
    def __init__(self, in_c, out_c=256):
        super().__init__()
        def block(ic, oc, s=1):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride=s, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            )
        self.s1 = block(in_c,  64, s=1)   # stride 1
        self.s2 = block(64,   128, s=2)   # stride 2  ← total stride 2
        self.s3 = block(128,  256, s=2)   # stride 4

        # FPN: upsample s3 → s2 resolution and concat
        self.up  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lat = nn.Conv2d(256, 128, 1, bias=False)
        self.neck = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)   # H/2, W/2
        f3 = self.s3(f2)   # H/4, W/4
        # Upsample f3 to f2 resolution and fuse
        feat = torch.cat([f2, self.up(self.lat(f3))], dim=1)  # (B, 256, H/2, W/2)
        return self.neck(feat)   # output stride 2 from input BEV


class PointPillarDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pfn   = PillarFeatureNet(in_dim=9, out_dim=cfg.pillar_feat_dim)
        self.backbone = BEVBackbone(in_c=cfg.pillar_feat_dim, out_c=256)
        # Per-class heatmap head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, cfg.num_classes, 1),
        )
        # Regression: dx,dy, log(w/wa), log(l/la), sin(yaw), cos(yaw), dz, log(h/ha)
        self.reg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 8 * cfg.num_classes, 1),   # per-class regression
        )
        self._init_weights()

    def _init_weights(self):
        pi = 0.01
        bias_val = -math.log((1 - pi) / pi)
        nn.init.constant_(self.heatmap_head[-1].bias, bias_val)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None and m is not self.heatmap_head[-1]:
                nn.init.constant_(m.bias, 0)

    def forward(self, pillars, pillar_idx, num_pts):
        """
        pillars:    (B, P_max, M, 9) — padded per-sample
        pillar_idx: (B, P_max)       — flattened BEV index
        num_pts:    (B, P_max)       — valid points per pillar
        """
        B = pillars.shape[0]
        bev_feats_list = []
        for b in range(B):
            P = (num_pts[b] > 0).sum()  # actual non-empty pillars
            pf = self.pfn(pillars[b, :P], num_pts[b, :P])      # (P, C)
            bev = scatter_to_bev(pf, pillar_idx[b, :P],
                                 self.cfg.bev_h, self.cfg.bev_w)   # (C, H, W)
            bev_feats_list.append(bev)
        bev_batch = torch.stack(bev_feats_list, dim=0)             # (B, C, H, W)

        feat = self.backbone(bev_batch)                            # (B, 256, H/2, W/2)
        heatmap = self.heatmap_head(feat)                          # (B, NC, H/2, W/2)
        reg = self.reg_head(feat)                                  # (B, NC*8, H/2, W/2)
        return heatmap, reg
```

---

## 4. Anchor-Based Regression (CRITICAL FIX)

Do NOT regress absolute box sizes. Regress **residuals relative to class anchors**.

```python
# SCoT – anchor residual regression
# Sequential:
#   step 1 → for each GT box, get class anchor size (wa, la, ha)
#   step 2 → compute: dt = (t - ta) / (diagonal_a)  for centres
#             dw = log(w / wa), dl = log(l / la), dh = log(h / ha)
#   step 3 → store in reg_map at class-specific channels: cls_idx*8 : cls_idx*8+8
# Branch:
#   at decode: multiply by anchor to recover absolute values
# Output: reg_map shape = (num_classes*8, out_h, out_w)

def build_regression_target(box, cls_idx, cfg, out_stride):
    """
    Returns 8-channel regression residual for anchor-based regression.
    box: dict with cx,cy,cz,w,l,h,yaw  (already in ego frame, ego coords)
    """
    cls_name = cfg.idx2class[cls_idx]
    wa, la, ha = cfg.anchor_sizes[cls_name]
    da = math.sqrt(wa**2 + la**2)   # BEV diagonal of anchor
    # Position offsets normalised by anchor diagonal
    # (centre in pixels decoded from heatmap peak; dx,dy are sub-pixel offsets)
    # Size: log residuals from anchor
    dw = math.log(max(box["w"], 0.01) / wa)
    dl = math.log(max(box["l"], 0.01) / la)
    dh = math.log(max(box["h"], 0.01) / ha)
    sin_yaw = math.sin(box["yaw"])
    cos_yaw = math.cos(box["yaw"])
    return [0.0, 0.0, dw, dl, sin_yaw, cos_yaw, box["cz"], dh]
    # channels 0,1 = sub-pixel offset dx,dy filled by caller


def decode_with_anchor(reg_slice, xi, yi, cls_idx, cfg, x_scale, y_scale):
    """
    Decode a single regression slice (8 values) for class cls_idx.
    reg_slice: tensor of shape (8,)
    Returns: dict of cx,cy,cz,w,l,h,yaw in ego frame
    """
    cls_name = cfg.idx2class[cls_idx]
    wa, la, ha = cfg.anchor_sizes[cls_name]
    dx, dy = reg_slice[0].item(), reg_slice[1].item()
    cx = (xi + dx) * x_scale + cfg.x_min
    cy = (yi + dy) * y_scale + cfg.y_min
    w  = wa * math.exp(reg_slice[2].item())
    l  = la * math.exp(reg_slice[3].item())
    sin_yaw, cos_yaw = reg_slice[4].item(), reg_slice[5].item()
    yaw = math.atan2(sin_yaw, cos_yaw)
    cz  = reg_slice[6].item()
    h   = ha * math.exp(reg_slice[7].item())
    return dict(cx=cx, cy=cy, cz=cz, w=w, l=l, h=h, yaw=yaw)
```

---

## 5. Rotated-NMS Implementation (CRITICAL FIX)

After decoding, apply proper BEV IoU NMS to remove duplicate detections.

```python
# SCoT – BEV Rotated NMS
# Sequential:
#   step 1 → compute BEV polygon for each detection (rotated rectangle)
#   step 2 → compute pairwise IoU via polygon intersection
#   step 3 → greedy NMS: keep highest-score box, suppress overlapping ones
# Branch:
#   if shapely available → use it; else use axis-aligned IoU as fallback
# Loop:
#   while detections remain in sorted list

def bev_poly_iou(b1, b2):
    """Rotated BEV IoU between two boxes using shapely (preferred) or AA fallback."""
    try:
        from shapely.geometry import Polygon
        def box_poly(b):
            cx, cy, w, l, yaw = b["cx"], b["cy"], b["w"], b["l"], b["yaw"]
            hw, hl = w / 2, l / 2
            corners = [(-hl,-hw),(hl,-hw),(hl,hw),(-hl,hw)]
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            rot = [(cx + cos_y*dx - sin_y*dy, cy + sin_y*dx + cos_y*dy)
                   for dx, dy in corners]
            return Polygon(rot)
        p1, p2 = box_poly(b1), box_poly(b2)
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return inter / (union + 1e-6)
    except ImportError:
        # Axis-aligned fallback
        ix1 = max(b1["cx"]-b1["l"]/2, b2["cx"]-b2["l"]/2)
        ix2 = min(b1["cx"]+b1["l"]/2, b2["cx"]+b2["l"]/2)
        iy1 = max(b1["cy"]-b1["w"]/2, b2["cy"]-b2["w"]/2)
        iy2 = min(b1["cy"]+b1["w"]/2, b2["cy"]+b2["w"]/2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        a1 = b1["l"] * b1["w"]
        a2 = b2["l"] * b2["w"]
        return inter / (a1 + a2 - inter + 1e-6)


def rotated_nms(detections, iou_thresh=0.2):
    """
    Greedy NMS on list of detection dicts (each must have 'score').
    Returns filtered list.
    """
    if len(detections) == 0:
        return []
    dets = sorted(detections, key=lambda x: x["score"], reverse=True)
    keep = []
    suppressed = [False] * len(dets)
    for i, d in enumerate(dets):
        if suppressed[i]:
            continue
        keep.append(d)
        for j in range(i+1, len(dets)):
            if suppressed[j]:
                continue
            if dets[j]["cls"] == d["cls"]:  # only suppress same class
                if bev_poly_iou(d, dets[j]) > iou_thresh:
                    suppressed[j] = True
    return keep
```

---

## 6. Test-Time Augmentation (MANDATORY, was missing entirely)

```python
# SCoT – TTA
# Sequential:
#   step 1 → define augmentation variants: original, flip_x, flip_y, rot_180
#   step 2 → for each variant: transform BEV pillars, run inference, decode dets
#   step 3 → inverse-transform decoded boxes back to ego frame
#   step 4 → merge all detections via rotated NMS (or WBF)
# Branch:
#   apply per-variant transform to point cloud before voxelization
# Loop:
#   for aug in ["orig", "flip_x", "flip_y", "rot180"]

def tta_augment_points(points, aug):
    """Apply augmentation to point cloud (N,3) for TTA."""
    pts = points.copy()
    if aug == "flip_x":
        pts[:, 0] = -pts[:, 0]
    elif aug == "flip_y":
        pts[:, 1] = -pts[:, 1]
    elif aug == "rot180":
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]
    return pts

def tta_inverse_box(det, aug):
    """Inverse-transform decoded box back to original frame."""
    d = dict(det)
    if aug == "flip_x":
        d["cx"] = -d["cx"]
        d["yaw"] = math.pi - d["yaw"]
    elif aug == "flip_y":
        d["cy"] = -d["cy"]
        d["yaw"] = -d["yaw"]
    elif aug == "rot180":
        d["cx"] = -d["cx"]
        d["cy"] = -d["cy"]
        d["yaw"] = d["yaw"] + math.pi
    return d

def tta_inference(model, points, cfg, device, score_thresh=0.1):
    """Run TTA and merge detections."""
    all_dets = []
    for aug in ["orig", "flip_x", "flip_y", "rot180"]:
        aug_pts = tta_augment_points(points, aug)
        pillars, pidx, npts = voxelize(aug_pts, cfg)
        # ... (pad to batch size 1 and move to device) ...
        with torch.no_grad():
            hm, reg = model(pillars_t, pidx_t, npts_t)
        dets_aug = decode_predictions(hm[0].cpu(), reg[0].cpu(), cfg, score_thresh)
        for d in dets_aug:
            all_dets.append(tta_inverse_box(d, aug))
    return rotated_nms(all_dets, iou_thresh=0.2)
```

---

## 7. Structured Reasoning Steps (SCoT)

```python
# Function: train_and_evaluate_3d_model(data_path)
# INPUT: str data_path
# OUTPUT: None (writes submission.csv)

# Sequential:
# 1. Load JSON metadata: sample_data, ego_pose, calibrated_sensor
# 2. Build token_to_info with sensor→ego calibration (s2e_R, s2e_t)
# 3. Parse training annotations from train.csv

# 4. Build PointPillarDetector (PillarFeatureNet + BEVBackbone + heads)

# 5. Dataset & DataLoader
#    - LyftPillarDataset.__getitem__:
#      a. load_lidar_in_ego_frame (apply s2e_R, s2e_t — CRITICAL)
#      b. GT-Augmentation (database sampling)
#      c. Geometric augmentation (flip x/y, random yaw ± π/4)
#      d. voxelize() → pillars, pillar_idx, num_pts (on CPU)
#      e. build_targets → per-class heatmap (stride=2) + anchor-residual reg_map

# 6. Training Loop
# Loop: epoch in range(60)
#   Loop: batch in train_loader
#     - Move pillars/idx/npts to GPU
#     - Forward → heatmap, reg
#     - Focal loss on heatmap (with heatmap bias init)
#     - L1 loss on reg residuals (only at positive cells)
#     - Backprop + OneCycleLR step
#   - Save best checkpoint by validation mAP (not by loss!)

# 7. Inference with TTA
# Loop: token in test_tokens
#   - load_lidar_in_ego_frame
#   - tta_inference (4 augmentations)
#   - rotated_nms(merged_dets, iou_thresh=0.2)
#   - local_to_world for each detection
#   - write to submission CSV with score ≥ 0.1 threshold
```

---

## 8. Additional Critical Details

### 8.1 GT-Augmentation Must Use Ego Frame
```python
# When building the GT database:
#   - Store point clouds already in ego frame (after sensor→ego transform)
#   - Crop points using EGO-frame box coordinates (after world_to_local)
#   - Paste back into ego-frame point cloud for current sample
```

### 8.2 Score Threshold Tuning
```python
# Use a LOW score threshold during inference: 0.05 to 0.1
# Then let NMS clean up duplicates.
# The competition metric rewards recall over precision at moderate thresholds.
# score_thresh = 0.1   # not 0.2 — 0.2 misses too many valid detections
```

### 8.3 Max Detections Per Frame
```python
# The competition allows many predictions per frame.
# Use max_det=500 per class per frame before NMS, then NMS reduces to ~50-100.
# Do NOT hard-cap at 100 before NMS — this prematurely discards high-recall candidates.
```

### 8.4 Model Selection
```python
# Save checkpoint based on validation mAP (actual 3D IoU metric), NOT validation loss.
# Compute val mAP at IoU thresholds [0.5, 0.55, 0.6, 0.65, 0.7] as a proxy
# (not 0.5-0.95 which is slow; 5 thresholds are sufficient for checkpoint selection).
# The model with best val mAP is the one to submit.
```

### 8.5 OneCycleLR (Required)
```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=cfg.lr,          # 2e-3
    epochs=cfg.num_epochs,   # 60
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=1e4,
)
# scheduler.step() called AFTER each batch (not after each epoch)
```

### 8.6 DataLoader Collate
```python
# voxelize() returns variable-length arrays (different P per sample).
# Use a custom collate_fn that pads pillars to max P in the batch:
def collate_fn(batch):
    max_P = max(b["pillars"].shape[0] for b in batch)
    # Pad each sample's pillars/idx/npts tensors to max_P
    # Return stacked batch tensors
    ...
```

---

## 9. Trigger Prompt

"Please implement a **Medal-Winning Top-10%** `runfile_0.py` for the Lyft 3D Object Detection dataset.

**MANDATORY architecture**: True PointPillars (PillarFeatureNet with mini-PointNet per pillar → scatter to pseudo-image → 2D backbone). **FORBIDDEN**: simple BEV multi-channel stacking (that scores only 0.017).

**MANDATORY data loading**: Read `calibrated_sensor.json`, apply sensor→ego transform (rotation + translation) to LiDAR points before use. Skipping this causes all detections to be spatially misaligned.

**MANDATORY inference**: TTA with flip_x, flip_y, rot_180 augmentations; inverse-transform decoded boxes; merge all via rotated BEV IoU NMS (use shapely). Score threshold = 0.1 (not 0.2).

**MANDATORY regression**: anchor-residual per class (log(w/wa), log(l/la), etc.) with class-specific anchor sizes. Output stride = 2 (not 4) for pedestrian resolution.

**MANDATORY training**: 60 epochs, OneCycleLR, batch_size=16, GT-Augmentation (Database Sampling). Save checkpoint by validation mAP (not loss).

Think step by step, write SCoT comments in code, and produce a single complete `runfile_0.py`or `runfile_1.py`."
