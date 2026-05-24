# Overlapped Fingerprint Separation — 本機 Ubuntu RTX 4090 訓練指南

## 硬體確認

| 項目 | 你的規格 | 此專案需求 | 判定 |
|------|---------|-----------|------|
| GPU | RTX 4090 (24 GB VRAM) | DualUNet base_ch=32 ≈ 7.8M params，128×128 輸入 | ✅ 綽綽有餘，VRAM 用量 < 4 GB |
| RAM | 64 GB | 資料集全載入 + DataLoader workers | ✅ 充足 |
| vCPU | 10 | DataLoader num_workers=4~8 | ✅ 足夠 |

> 這個模型非常輕量（~7.8M 參數、128×128 灰階輸入），RTX 4090 跑這個相當於殺雞用牛刀。你可以考慮把 `batch_size` 從 16 提高到 64 甚至 128，充分利用 GPU。

---

## Step 0：環境前置

```bash
# 確認 NVIDIA driver 已安裝
nvidia-smi

# 預期輸出應包含:
#   Driver Version: 5xx.xx 以上
#   CUDA Version: 12.x
#   GeForce RTX 4090
```

如果 `nvidia-smi` 沒有輸出或出錯：

```bash
sudo apt update
sudo apt install -y nvidia-driver-550   # 或最新穩定版
sudo reboot
```

---

## Step 1：建立 Python 虛擬環境

建議用 conda 或 venv，避免污染系統 Python：

### 方案 A：conda（推薦）

```bash
# 安裝 miniconda（若尚未安裝）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# 建立環境
conda create -n fingerprint python=3.10 -y
conda activate fingerprint
```

### 方案 B：venv

```bash
sudo apt install -y python3.10 python3.10-venv python3-pip
python3.10 -m venv ~/fingerprint-env
source ~/fingerprint-env/bin/activate
```

---

## Step 2：安裝套件

```bash
# PyTorch with CUDA 12.x（RTX 4090 需要 CUDA 12+）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 其餘相依
pip install numpy opencv-python-headless scikit-image scikit-learn \
            matplotlib scipy tqdm jupyter ipykernel
```

驗證 CUDA 可用：

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
# 預期: CUDA: True, Device: NVIDIA GeForce RTX 4090
```

---

## Step 3：準備資料集

Notebook 支援 SOCOFing、FVC、NIST SD4 等格式。你需要在本機建立資料目錄：

```bash
mkdir -p ~/fingerprint-project/data/fingerprints
```

### 資料集選項（依大小排序）

| 資料集 | 張數 | 取得方式 | 建議 |
|--------|------|---------|------|
| SOCOFing | ~6,000 | Kaggle 下載 `socofing` | ✅ 最推薦，finger-level split 有足夠統計量 |
| FVC2000/2002/2004 | 800~3,200 | FVC 官方或 Kaggle | ✅ 經典指紋資料集 |
| NIST SD4 | 4,000 | NIST 官方申請 | 需要申請，品質最好 |

下載後把所有指紋影像（.BMP / .png / .jpg / .tif）放進 `~/fingerprint-project/data/fingerprints/` 資料夾下（可有子目錄）。

```bash
# 範例：從 Kaggle 下載 SOCOFing
pip install kaggle
kaggle datasets download -d ruizgara/socofing
unzip socofing.zip -d ~/fingerprint-project/data/fingerprints/
```

---

## Step 4：轉換 Notebook 為本機可執行

原始 notebook 使用 Kaggle 路徑，需要改兩個地方：

### 4a. 修改資料路徑

將 notebook 中的：
```python
BASE = "/kaggle/input/fingerprints"
```
改為：
```python
BASE = os.path.expanduser("~/fingerprint-project/data/fingerprints")
```

### 4b. 修改輸出路徑

將所有 `/kaggle/working/` 改為本機路徑：

```python
OUTPUT_DIR = os.path.expanduser("~/fingerprint-project/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

然後把所有 `/kaggle/working/xxx` 替換成 `os.path.join(OUTPUT_DIR, "xxx")`。

### 4c. 一鍵替換指令

```bash
cd ~/fingerprint-project

# 複製 notebook
cp /path/to/fingerprint_separation_final.ipynb ./train.ipynb

# 批次替換路徑
sed -i 's|/kaggle/input/fingerprints|'$HOME'/fingerprint-project/data/fingerprints|g' train.ipynb
sed -i 's|/kaggle/working/|'$HOME'/fingerprint-project/output/|g' train.ipynb

mkdir -p ~/fingerprint-project/output
```

---

## Step 5：調優參數（善用 RTX 4090）

原始設定是為 Kaggle 免費 GPU（T4 16GB）設計的。RTX 4090 可以提升：

```python
# 原始（為 Kaggle T4 設計）
IMG_SIZE, BATCH_SIZE = 128, 16
EPOCHS = 40
base_ch = 32
num_workers = 2

# 推薦（RTX 4090 最佳化）
IMG_SIZE, BATCH_SIZE = 128, 64       # batch 加大 4 倍，收斂更穩
EPOCHS = 60                          # 資料集大時可多跑幾輪
base_ch = 64                         # 通道加倍 → 模型容量更大
num_workers = 8                      # 充分利用 10 vCPU
```

> **VRAM 估算**：base_ch=64 + batch=64 + IMG_SIZE=128 → 約 8~10 GB VRAM，RTX 4090 的 24 GB 仍有大量餘裕。如果想要更高解析度（如 256×256），把 batch 降回 32 即可。

---

## Step 6：執行訓練

### 方式 A：Jupyter Notebook（互動觀察）

```bash
cd ~/fingerprint-project
jupyter notebook train.ipynb
```

### 方式 B：轉成 .py 腳本（推薦，適合長時間訓練）

```bash
# 轉換
jupyter nbconvert --to script train.ipynb --output train

# 用 nohup 或 tmux 背景執行
tmux new -s train
cd ~/fingerprint-project
python train.py 2>&1 | tee output/training.log
# Ctrl+B, D 離開 tmux（訓練繼續跑）
# tmux attach -t train  重新接回
```

### 方式 C：直接用下面的一體化腳本

---

## Step 7：一體化訓練腳本

我已將 notebook 關鍵邏輯整理成以下獨立 `.py`，你可以直接用：

```bash
python train_local.py \
    --data_dir ~/fingerprint-project/data/fingerprints \
    --output_dir ~/fingerprint-project/output \
    --batch_size 64 \
    --epochs 60 \
    --base_ch 64 \
    --num_workers 8 \
    --img_size 128
```

> 腳本見附件 `train_local.py`。

---

## Step 8：監控訓練

### GPU 使用監控

```bash
# 即時監控（每 1 秒更新）
watch -n1 nvidia-smi

# 或用 nvitop（更美觀）
pip install nvitop
nvitop
```

### 預期訓練時間

| 設定 | 資料集 | 每 Epoch | 總時間 |
|------|--------|---------|--------|
| batch=16, ch=32 | SOCOFing 6K | ~15 秒 | ~10 分鐘 |
| batch=64, ch=64 | SOCOFing 6K | ~8 秒 | ~8 分鐘 |
| batch=64, ch=64 | 大型合併集 20K+ | ~30 秒 | ~30 分鐘 |

> RTX 4090 跑這種 128×128 灰階小模型非常快。

---

## Step 9：訓練完成後

產出檔案在 `~/fingerprint-project/output/`：

```
output/
├── best_model.pth          # 最佳模型權重
├── full_report.json        # 完整評估報告
├── training_curve.png      # 訓練曲線
├── multi_protocol.png      # EER 分佈圖
├── id_vs_ood.png           # OOD 分析
├── cmc.png                 # CMC 曲線
├── worst5.png              # 最差 5 例
├── best5.png               # 最佳 5 例
└── training.log            # 訓練日誌
```

---

## 常見問題排除

### Q1：`CUDA out of memory`
→ 降低 `batch_size`（64→32→16），或降低 `base_ch`（64→32）

### Q2：`RuntimeError: DataLoader worker exited unexpectedly`
→ 降低 `num_workers`（8→4→2→0）

### Q3：`cv2.imread returns None`
→ 影像路徑有中文或特殊字元。確認圖片格式和路徑正確：
```bash
find ~/fingerprint-project/data/fingerprints -type f | head -20
```

### Q4：訓練 loss 不降
→ 檢查資料集是否正確載入（`找到 X 張指紋影像` 的數字是否 >0）
→ 確認影像是灰階指紋，不是其他格式

### Q5：想用 mixed precision 加速
```python
# 在訓練迴圈加入：
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    pred_a, pred_b = model(mix)
    loss, comp = criterion(pred_a, pred_b, gt_a, gt_b, mix, alpha)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
→ 可再加速 30~50%，VRAM 省一半。

---

## 完整指令速查

```bash
# 一次性完整流程
conda create -n fingerprint python=3.10 -y && conda activate fingerprint
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy opencv-python-headless scikit-image scikit-learn matplotlib scipy tqdm

mkdir -p ~/fingerprint-project/{data/fingerprints,output}
# 把指紋影像放進 data/fingerprints/

cd ~/fingerprint-project
# 用 train_local.py 或轉換後的 notebook 執行
python train_local.py --data_dir data/fingerprints --output_dir output
```
