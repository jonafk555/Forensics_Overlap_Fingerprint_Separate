#!/usr/bin/env python3
"""
Overlapped Fingerprint Separation — 本機訓練腳本
從 fingerprint_separation_final.ipynb 轉換而來，適用於 Ubuntu + RTX 4090

用法:
    python train_local.py \
        --data_dir ~/fingerprint-project/data/fingerprints \
        --output_dir ~/fingerprint-project/output \
        --batch_size 64 --epochs 60 --base_ch 64 --num_workers 8
"""

import os, sys, glob, random, cv2, json, re, argparse, time
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")  # 無 GUI 環境
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as nd_rotate
from tqdm import tqdm


# ============================================================
# CLI 參數
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Overlapped Fingerprint Separation Training")
    p.add_argument("--data_dir",    type=str, required=True,  help="指紋影像資料夾路徑")
    p.add_argument("--output_dir",  type=str, default="./output", help="輸出路徑")
    p.add_argument("--img_size",    type=int, default=128,    help="影像大小 (預設 128)")
    p.add_argument("--batch_size",  type=int, default=64,     help="Batch size (RTX 4090 建議 64)")
    p.add_argument("--epochs",      type=int, default=60,     help="訓練 epochs")
    p.add_argument("--base_ch",     type=int, default=64,     help="U-Net 基礎通道數 (32 或 64)")
    p.add_argument("--num_workers", type=int, default=8,      help="DataLoader workers")
    p.add_argument("--lr",          type=float, default=1e-4, help="學習率")
    p.add_argument("--seed",        type=int, default=42,     help="隨機種子")
    p.add_argument("--amp",         action="store_true",      help="啟用 mixed precision (AMP)")
    p.add_argument("--resume",      type=str, default=None,   help="從 checkpoint 繼續訓練")
    return p.parse_args()


# ============================================================
# 資料載入 + Finger-level Split
# ============================================================
def load_gray(path, size=128):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取影像: {path}")
    img = cv2.resize(img, (size, size))
    return img


def extract_subject_id(path):
    name = os.path.basename(path)
    m = re.match(r'^([A-Za-z]?\d+)', name)
    if m:
        return m.group(1)
    return os.path.basename(os.path.dirname(path))


def extract_finger_id(path):
    name = os.path.basename(path)
    m = re.match(r'^(\d+)__\w+?_(Left|Right)_(\w+)_finger', name)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    m = re.match(r'^(\d+)_(\d+)\.', name)
    if m:
        return m.group(1) + "_" + m.group(2)
    return extract_subject_id(path)


def load_and_split(data_dir, seed=42):
    all_imgs = []
    for ext in ["*.BMP", "*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        all_imgs.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    all_imgs = sorted(all_imgs)
    print(f"找到 {len(all_imgs)} 張指紋影像")

    if len(all_imgs) == 0:
        print(f"\n❌ 錯誤: 在 {data_dir} 找不到任何影像。")
        print(f"   請確認路徑正確，且包含 .BMP/.png/.jpg/.tif 影像。")
        print(f"   目錄內容:")
        for item in os.listdir(data_dir)[:10]:
            print(f"     {item}")
        sys.exit(1)

    n_subjects = len(set(extract_subject_id(p) for p in all_imgs))
    n_fingers  = len(set(extract_finger_id(p)  for p in all_imgs))
    print(f"\n[Dataset 統計]")
    print(f"  Total images:  {len(all_imgs)}")
    print(f"  Subjects:      {n_subjects}")
    print(f"  Finger units:  {n_fingers}")
    print(f"  Imgs/finger:   {len(all_imgs) / max(n_fingers, 1):.1f}")

    if n_fingers >= 100:
        extract_id = extract_finger_id
        split_strategy = "finger-level"
    else:
        extract_id = extract_subject_id
        split_strategy = "subject-level"
    print(f"  Split strategy: {split_strategy}")

    id_to_paths = defaultdict(list)
    for p in all_imgs:
        id_to_paths[extract_id(p)].append(p)

    all_ids = sorted(id_to_paths.keys())
    random.Random(seed).shuffle(all_ids)
    n_id = len(all_ids)
    train_ids = set(all_ids[:int(n_id * 0.70)])
    val_ids   = set(all_ids[int(n_id * 0.70):int(n_id * 0.85)])
    test_ids  = set(all_ids[int(n_id * 0.85):])

    train_paths = [p for i in train_ids for p in id_to_paths[i]]
    val_paths   = [p for i in val_ids   for p in id_to_paths[i]]
    test_paths  = [p for i in test_ids  for p in id_to_paths[i]]

    # Leakage 驗證
    ids_train = set(extract_id(p) for p in train_paths)
    ids_test  = set(extract_id(p) for p in test_paths)
    ids_val   = set(extract_id(p) for p in val_paths)
    assert len(ids_train & ids_test) == 0, "Train/Test leakage detected!"
    assert len(ids_train & ids_val) == 0,  "Train/Val leakage detected!"

    print(f"\n[Split 結果]")
    print(f"  Train: {len(train_paths):5d} 張 / {len(train_ids):4d} ids")
    print(f"  Val:   {len(val_paths):5d} 張 / {len(val_ids):4d} ids")
    print(f"  Test:  {len(test_paths):5d} 張 / {len(test_ids):4d} ids")
    print(f"  ✓ 無 leakage")

    if len(test_paths) < 100:
        print(f"\n  ⚠️ Test set 只有 {len(test_paths)} 張，評估統計力不足。")
    elif len(test_paths) < 500:
        print(f"\n  ⚠ Test set {len(test_paths)} 張，評估可信但 CI 會較寬")
    else:
        print(f"\n  ✓ Test set 充足，評估統計力良好")

    return (all_imgs, train_paths, val_paths, test_paths,
            n_subjects, n_fingers, split_strategy,
            ids_train, ids_val, ids_test, extract_id)


# ============================================================
# 合成 + Dataset
# ============================================================
def synthesize_overlap(img_a, img_b, alpha=None, angle=None,
                        tx=None, ty=None, randomize_roles=False):
    if randomize_roles and random.random() < 0.5:
        img_a, img_b = img_b, img_a

    H, W = img_a.shape
    if alpha is None: alpha = random.uniform(0.35, 0.65)
    if angle is None: angle = random.uniform(-25, 25)
    if tx is None:    tx = random.randint(-20, 20)
    if ty is None:    ty = random.randint(-20, 20)

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img_b_t = cv2.warpAffine(img_b, M, (W, H), borderValue=255)
    img_b_t = nd_rotate(img_b_t, angle, reshape=False, cval=255)
    img_b_t = np.clip(img_b_t, 0, 255).astype(np.uint8)

    img_mix = alpha * img_a.astype(np.float32) + (1 - alpha) * img_b_t.astype(np.float32)
    img_mix = np.clip(img_mix, 0, 255).astype(np.uint8)
    return img_mix, img_a, img_b_t, alpha


class OverlapDataset(Dataset):
    def __init__(self, paths, size=128, randomize_roles=False):
        self.paths = paths
        self.size  = size
        self.randomize_roles = randomize_roles

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path_a = self.paths[idx]
        j = random.randint(0, len(self.paths) - 1)
        while j == idx and len(self.paths) > 1:
            j = random.randint(0, len(self.paths) - 1)
        path_b = self.paths[j]

        img_a = load_gray(path_a, self.size)
        img_b = load_gray(path_b, self.size)
        mix, gt_a, gt_b, alpha = synthesize_overlap(
            img_a, img_b, randomize_roles=self.randomize_roles)

        def to_tensor(arr):
            return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)

        return (to_tensor(mix), to_tensor(gt_a), to_tensor(gt_b),
                torch.tensor(alpha, dtype=torch.float32))


# ============================================================
# Dual U-Net
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))
    def forward(self, x): return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class DualUNet(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(1, base_ch)
        self.enc2 = Down(base_ch,   base_ch*2)
        self.enc3 = Down(base_ch*2, base_ch*4)
        self.enc4 = Down(base_ch*4, base_ch*8)
        self.bottleneck = Down(base_ch*8, base_ch*16)
        self.up4a = Up(base_ch*16, base_ch*8, base_ch*8)
        self.up3a = Up(base_ch*8,  base_ch*4, base_ch*4)
        self.up2a = Up(base_ch*4,  base_ch*2, base_ch*2)
        self.up1a = Up(base_ch*2,  base_ch,   base_ch)
        self.out_a = nn.Sequential(nn.Conv2d(base_ch, 1, 1), nn.Sigmoid())
        self.up4b = Up(base_ch*16, base_ch*8, base_ch*8)
        self.up3b = Up(base_ch*8,  base_ch*4, base_ch*4)
        self.up2b = Up(base_ch*4,  base_ch*2, base_ch*2)
        self.up1b = Up(base_ch*2,  base_ch,   base_ch)
        self.out_b = nn.Sequential(nn.Conv2d(base_ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        e4 = self.enc4(e3); b  = self.bottleneck(e4)
        a = self.up4a(b,  e4); a = self.up3a(a, e3); a = self.up2a(a, e2); a = self.up1a(a, e1)
        bb = self.up4b(b, e4); bb = self.up3b(bb, e3); bb = self.up2b(bb, e2); bb = self.up1b(bb, e1)
        return self.out_a(a), self.out_b(bb)


# ============================================================
# Loss
# ============================================================
def gaussian_window(window_size=11, sigma=1.5, channels=1):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", gaussian_window(window_size, sigma, 1))
        self.C1 = 0.01 ** 2; self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        w = self.window.to(pred.device); pad = self.window_size // 2
        mu_p = F.conv2d(pred,   w, padding=pad)
        mu_t = F.conv2d(target, w, padding=pad)
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sig_p2 = F.conv2d(pred * pred,     w, padding=pad) - mu_p2
        sig_t2 = F.conv2d(target * target, w, padding=pad) - mu_t2
        sig_pt = F.conv2d(pred * target,   w, padding=pad) - mu_pt
        ssim_map = ((2 * mu_pt + self.C1) * (2 * sig_pt + self.C2)) / \
                   ((mu_p2 + mu_t2 + self.C1) * (sig_p2 + sig_t2 + self.C2))
        return 1.0 - ssim_map.mean()


class FixedLoss(nn.Module):
    def __init__(self, lambda_mix=0.5, lambda_div=0.3, lambda_ssim=0.3,
                 div_margin=0.20):
        super().__init__()
        self.l1 = nn.L1Loss(); self.ssim_loss = SSIMLoss()
        self.lambda_mix = lambda_mix
        self.lambda_div = lambda_div
        self.lambda_ssim = lambda_ssim
        self.div_margin = div_margin

    def _pair_loss(self, p, g):
        return self.l1(p, g) + self.lambda_ssim * self.ssim_loss(p, g)

    def forward(self, pred_a, pred_b, gt_a, gt_b, mix, alpha):
        B = pred_a.size(0)
        loss_recon = 0.0
        for i in range(B):
            l_aa = self._pair_loss(pred_a[i:i+1], gt_a[i:i+1]) + \
                   self._pair_loss(pred_b[i:i+1], gt_b[i:i+1])
            l_ab = self._pair_loss(pred_a[i:i+1], gt_b[i:i+1]) + \
                   self._pair_loss(pred_b[i:i+1], gt_a[i:i+1])
            loss_recon = loss_recon + torch.min(l_aa, l_ab)
        loss_recon = loss_recon / B

        alpha_b = alpha.view(-1, 1, 1, 1).to(pred_a.device)
        recon_mix = alpha_b * pred_a + (1 - alpha_b) * pred_b
        loss_mix = self.l1(recon_mix, mix)

        diff = (pred_a - pred_b).abs().mean()
        loss_div = F.relu(self.div_margin - diff)

        total = loss_recon + self.lambda_mix * loss_mix + self.lambda_div * loss_div
        return total, {"recon": loss_recon.item(), "mix": loss_mix.item(),
                       "div": loss_div.item(), "total": total.item()}


# ============================================================
# 評估工具
# ============================================================
def to_t(arr, device):
    return torch.from_numpy(arr.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)


def best_pit_assignment(pa, pb, ga, gb):
    s_aa = ssim(pa, ga, data_range=255); s_bb = ssim(pb, gb, data_range=255)
    s_ab = ssim(pa, gb, data_range=255); s_ba = ssim(pb, ga, data_range=255)
    if (s_aa + s_bb) >= (s_ab + s_ba):
        return pa, pb, False, (s_aa, s_bb)
    return pb, pa, True, (s_ab, s_ba)


def eer_from_scores(scores_genuine, scores_impostor):
    labels = [1]*len(scores_genuine) + [0]*len(scores_impostor)
    scores = list(scores_genuine) + list(scores_impostor)
    if len(set(scores)) < 2:
        return 1.0, 0.5, 0.0, None
    try:
        fpr, tpr, thr = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        auc = roc_auc_score(labels, scores)
        rank1 = sum(1 for s in scores_genuine if s >= thr[eer_idx]) / len(scores_genuine)
        return eer, auc, rank1, thr[eer_idx]
    except Exception:
        return 1.0, 0.5, 0.0, None


def bootstrap_ci(values, n_boot=1000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boots, (100-ci)/2)
    hi = np.percentile(boots, 100 - (100-ci)/2)
    return float(np.mean(arr)), float(lo), float(hi)


def bootstrap_eer(gen, imp, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    gen, imp = np.array(gen), np.array(imp)
    eers = []
    for _ in range(n_boot):
        g_idx = rng.integers(0, len(gen), len(gen))
        i_idx = rng.integers(0, len(imp), len(imp))
        eer, _, _, _ = eer_from_scores(gen[g_idx].tolist(), imp[i_idx].tolist())
        eers.append(eer)
    return (float(np.mean(eers)),
            float(np.percentile(eers, 2.5)),
            float(np.percentile(eers, 97.5)))


def collect_outputs(model, paths, img_size, device, n_samples=200,
                    alpha=0.5, angle=15, tx=10, ty=10, seed=0):
    model.eval()
    rng = random.Random(seed)
    indices = list(range(len(paths)))
    rng.shuffle(indices)
    n_samples = min(n_samples, len(indices))
    outputs = []
    with torch.no_grad():
        for idx_a in tqdm(indices[:n_samples], desc="Collecting outputs"):
            idx_b = rng.choice(indices)
            while idx_b == idx_a and len(indices) > 1:
                idx_b = rng.choice(indices)
            img_a = load_gray(paths[idx_a], img_size)
            img_b = load_gray(paths[idx_b], img_size)
            mix_np, gt_a, gt_b, _ = synthesize_overlap(
                img_a, img_b, alpha=alpha, angle=angle, tx=tx, ty=ty,
                randomize_roles=False)
            pred_a, pred_b = model(to_t(mix_np, device))
            pa = (pred_a.cpu().squeeze().numpy() * 255).astype(np.uint8)
            pb = (pred_b.cpu().squeeze().numpy() * 255).astype(np.uint8)
            pa_aln, pb_aln, sw, (s_a, s_b) = best_pit_assignment(
                pa, pb, gt_a.astype(np.uint8), gt_b.astype(np.uint8))
            outputs.append({
                "idx_a": idx_a, "idx_b": idx_b, "mix": mix_np,
                "gt_a": gt_a.astype(np.uint8), "gt_b": gt_b.astype(np.uint8),
                "pred_for_a": pa_aln, "pred_for_b": pb_aln, "swapped": sw,
                "ssim_genuine_a": s_a, "ssim_genuine_b": s_b,
            })
    return outputs


# ============================================================
# 主程式
# ============================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("⚠ 未偵測到 CUDA GPU，使用 CPU（會非常慢）")

    # ── 資料載入 ──
    print("\n" + "="*70)
    print("Step 1: 資料載入與切分")
    print("="*70)
    (all_imgs, train_paths, val_paths, test_paths,
     n_subjects, n_fingers, split_strategy,
     ids_train, ids_val, ids_test, extract_id) = load_and_split(args.data_dir, args.seed)

    # ── DataLoader ──
    IMG_SIZE = args.img_size
    train_loader = DataLoader(
        OverlapDataset(train_paths, IMG_SIZE, randomize_roles=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        OverlapDataset(val_paths, IMG_SIZE, randomize_roles=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    print(f"\nDataLoaders ready (batch={args.batch_size}, workers={args.num_workers})")

    # ── 模型 ──
    print("\n" + "="*70)
    print("Step 2: 建立模型")
    print("="*70)
    model = DualUNet(base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {n_params:,}")
    print(f"基礎通道:   {args.base_ch}")

    # 估算 VRAM
    param_mem = n_params * 4 / 1e9  # float32
    print(f"參數記憶體: {param_mem:.2f} GB (不含 optimizer state 和 activation)")

    # Resume
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"✓ 從 epoch {start_epoch} 繼續訓練")

    # ── Loss & Optimizer ──
    criterion = FixedLoss(lambda_mix=0.5, lambda_div=0.3,
                          lambda_ssim=0.3, div_margin=0.20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP
    scaler = torch.amp.GradScaler('cuda') if args.amp and device == "cuda" else None
    if scaler:
        print("✓ Mixed Precision (AMP) 已啟用")

    # ── 訓練 ──
    print("\n" + "="*70)
    print(f"Step 3: 開始訓練 ({args.epochs} epochs)")
    print("="*70)
    EPOCHS = args.epochs
    history = {"train_loss": [], "val_loss": [],
               "train_recon": [], "train_mix": [], "train_div": []}
    best_val = float("inf")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tl, tr, tm, td = 0.0, 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for mix, gt_a, gt_b, alpha in pbar:
            mix, gt_a, gt_b, alpha = (mix.to(device), gt_a.to(device),
                                       gt_b.to(device), alpha.to(device))
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    pred_a, pred_b = model(mix)
                    loss, comp = criterion(pred_a, pred_b, gt_a, gt_b, mix, alpha)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_a, pred_b = model(mix)
                loss, comp = criterion(pred_a, pred_b, gt_a, gt_b, mix, alpha)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            tl += comp["total"]; tr += comp["recon"]
            tm += comp["mix"];   td += comp["div"]
            pbar.set_postfix(loss=f"{comp['total']:.4f}")

        N = len(train_loader)
        tl, tr, tm, td = tl/N, tr/N, tm/N, td/N

        # Validation
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for mix, gt_a, gt_b, alpha in val_loader:
                mix, gt_a, gt_b, alpha = (mix.to(device), gt_a.to(device),
                                           gt_b.to(device), alpha.to(device))
                if scaler:
                    with torch.amp.autocast('cuda'):
                        pred_a, pred_b = model(mix)
                        _, comp = criterion(pred_a, pred_b, gt_a, gt_b, mix, alpha)
                else:
                    pred_a, pred_b = model(mix)
                    _, comp = criterion(pred_a, pred_b, gt_a, gt_b, mix, alpha)
                vl += comp["total"]
        vl /= len(val_loader)
        scheduler.step()

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_recon"].append(tr); history["train_mix"].append(tm)
        history["train_div"].append(td)

        elapsed = time.time() - t0
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0

        if vl < best_val:
            best_val = vl
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
            }, os.path.join(args.output_dir, "best_model.pth"))
            marker = " ★ best"
        else:
            marker = ""

        print(f"E{epoch:3d} | Train: {tl:.4f} (r={tr:.4f} m={tm:.4f} d={td:.4f}) "
              f"| Val: {vl:.4f} | {elapsed:.1f}s | GPU: {gpu_mem:.1f}GB{marker}")

    # 訓練曲線
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["train_recon"], label="Recon (PIT)")
    axes[1].plot(history["train_mix"],   label="Mixture Consistency")
    axes[1].plot(history["train_div"],   label="Diversity Hinge")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Component Loss")
    axes[1].set_title("Loss Components"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curve.png"), dpi=150, bbox_inches="tight")
    print(f"\n✓ 訓練完成，Best Val Loss = {best_val:.4f}")

    # 載入最佳模型
    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── 評估 ──
    print("\n" + "="*70)
    print("Step 4: 模型評估")
    print("="*70)
    N_MAIN = min(300, max(50, len(test_paths) * 2))
    print(f"評估樣本數: n={N_MAIN}")

    outputs_main = collect_outputs(model, test_paths, IMG_SIZE, device,
                                   n_samples=N_MAIN, alpha=0.5, angle=15)

    # Multi-protocol EER
    def compute_multi_protocol(outputs):
        gen = [o["ssim_genuine_a"] for o in outputs]
        n = len(outputs)
        rng1 = random.Random(123); rng2 = random.Random(456); rng3 = random.Random(789)
        imp_p1 = [ssim(o["pred_for_a"], o["gt_b"], data_range=255) for o in outputs]
        imp_p2 = []
        for i in range(n):
            j = rng1.choice([k for k in range(n) if k != i])
            imp_p2.append(ssim(outputs[i]["pred_for_a"], outputs[j]["gt_a"], data_range=255))
        imp_p3 = []
        for i in range(n):
            j = rng2.choice([k for k in range(n) if k != i])
            imp_p3.append(ssim(outputs[i]["pred_for_a"], outputs[j]["gt_a"], data_range=255))
        imp_p4 = []
        for i in range(n):
            j = rng3.choice([k for k in range(n) if k != i])
            imp_p4.append(ssim(outputs[i]["pred_for_a"], outputs[j]["pred_for_a"], data_range=255))
        return {
            "P1_within_mix":   eer_from_scores(gen, imp_p1),
            "P2_random_raw":   eer_from_scores(gen, imp_p2),
            "P3_cross_case":   eer_from_scores(gen, imp_p3),
            "P4_pred_vs_pred": eer_from_scores(gen, imp_p4),
        }, {"gen": gen, "imp_p1": imp_p1, "imp_p2": imp_p2,
            "imp_p3": imp_p3, "imp_p4": imp_p4}

    multi_eer, score_dists = compute_multi_protocol(outputs_main)

    print(f"\n{'Protocol':<20} {'EER% (95% CI)':<28} {'AUC':>8} {'Rank-1 %':>10}")
    print("-" * 72)
    eer_with_ci = {}
    for name, (eer, auc, r1, _) in multi_eer.items():
        imp_key = {"P1_within_mix": "imp_p1", "P2_random_raw": "imp_p2",
                   "P3_cross_case": "imp_p3", "P4_pred_vs_pred": "imp_p4"}[name]
        eer_m, eer_lo, eer_hi = bootstrap_eer(score_dists["gen"], score_dists[imp_key])
        eer_with_ci[name] = (eer, eer_lo, eer_hi, auc, r1)
        print(f"{name:<20} {eer*100:5.2f}  [{eer_lo*100:5.2f}, {eer_hi*100:5.2f}]   "
              f"{auc:>7.4f}  {r1*100:>9.2f}")
    print("-" * 72)

    # ΔSSIM
    ssim_vals = [o["ssim_genuine_a"] for o in outputs_main]
    base_ssims = [ssim(o["mix"], o["gt_a"], data_range=255) for o in outputs_main]
    delta_vals = [a - b for a, b in zip(ssim_vals, base_ssims)]
    m_d, lo_d, hi_d = bootstrap_ci(delta_vals)
    sig = "✓ 顯著" if lo_d > 0.05 else "⚠ 不顯著"
    print(f"\nΔSSIM: {m_d:+.4f} [{lo_d:+.4f}, {hi_d:+.4f}]  {sig}")

    # 存檔
    leakage_clean = (len(ids_train & ids_test) == 0 and len(ids_train & ids_val) == 0)
    report = {
        "dataset": {
            "n_images": len(all_imgs), "n_subjects": n_subjects,
            "n_finger_units": n_fingers, "split_strategy": split_strategy,
            "n_train": len(train_paths), "n_val": len(val_paths), "n_test": len(test_paths),
        },
        "training": {
            "epochs": EPOCHS, "batch_size": args.batch_size,
            "img_size": IMG_SIZE, "base_ch": args.base_ch,
            "best_val_loss": best_val, "n_params": n_params, "amp": args.amp,
        },
        "leakage_clean": leakage_clean,
        "multi_protocol_eer": {
            name: {"eer": float(e), "ci_lo": float(lo), "ci_hi": float(hi),
                   "auc": float(auc), "rank1": float(r1)}
            for name, (e, lo, hi, auc, r1) in eer_with_ci.items()
        },
        "delta_ssim": {"mean": m_d, "ci_lo": lo_d, "ci_hi": hi_d, "significant": lo_d > 0.05},
        "hardware": {
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            "vram_gb": torch.cuda.get_device_properties(0).total_mem / 1e9 if device == "cuda" else 0,
        }
    }

    report_path = os.path.join(args.output_dir, "full_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"訓練與評估完成")
    print(f"{'='*70}")
    print(f"  模型權重:   {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"  評估報告:   {report_path}")
    print(f"  訓練曲線:   {os.path.join(args.output_dir, 'training_curve.png')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
