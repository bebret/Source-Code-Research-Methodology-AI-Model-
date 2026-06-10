# ====================================================================
# SCRIPT MASTER FINAL: EFFICIENTNET-V2-S (RESOLUSI 640) - FIXED
# ====================================================================
!pip install -q roboflow scikit-learn matplotlib pandas tqdm

from google.colab import drive
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast   # ← AMP untuk kecepatan
from roboflow import Roboflow
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# 1. SETUP GOOGLE DRIVE
drive.mount('/content/drive')
drive_path = "/content/drive/MyDrive/Skripsi_EfficientNetV2S_640"
os.makedirs(drive_path, exist_ok=True)

# 2. DOWNLOAD DATASET
print("\n⏳ Mengunduh Dataset dari Roboflow...")
rf = Roboflow(api_key="CU04tyVnI9U4FqyF9aPv")
project = rf.workspace("alberts-workspace-lrb7z").project("skin-disease-3-jjow2")
version = project.version(3)
dataset = version.download("folder")

# 3. PERSIAPAN DATASET (640px)
print("\n⚙️ Menyiapkan Dataset & Augmentasi (640px)...")
IMG_SIZE = 640
BATCH_SIZE = 8

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_dir = dataset.location
train_dir = os.path.join(base_dir, "train")

# ✅ FIX #2: Validasi val_dir tidak None
val_dir = next((os.path.join(base_dir, d) for d in ["valid", "val", "test"]
                if os.path.exists(os.path.join(base_dir, d))), None)
if val_dir is None:
    raise FileNotFoundError("❌ Folder validasi tidak ditemukan! Cek struktur dataset.")

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data   = datasets.ImageFolder(val_dir,   transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

kelas_penyakit = train_data.classes
num_classes    = len(kelas_penyakit)
print(f"✅ Jumlah kelas: {num_classes} → {kelas_penyakit}")

# 4. INISIALISASI MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Device: {device}")

model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

# ✅ FIX #1: Akses classifier[1], bukan .in_features langsung
jumlah_saraf_input   = model.classifier[1].in_features   # = 1280
model.classifier[1]  = nn.Linear(jumlah_saraf_input, num_classes)

model     = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scaler    = GradScaler()   # ← untuk AMP

last_ckpt_path = f"{drive_path}/last_effv2s.pth"
best_ckpt_path = f"{drive_path}/best_effv2s.pth"
start_epoch = 0
best_acc    = 0.0

# AUTO-RESUME
if os.path.exists(last_ckpt_path):
    print("\n🔄 Melanjutkan training dari Drive...")
    # ✅ FIX #3: Tambah map_location=device
    checkpoint = torch.load(last_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc    = checkpoint['best_acc']
    print(f"   Lanjut dari Epoch {start_epoch}, Best Acc sebelumnya: {best_acc:.2f}%")
else:
    print("\n🚀 MEMULAI TRAINING BARU (EPOCH 1)...")

# 5. TRAINING LOOP
TOTAL_EPOCHS = 35

for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    print(f"\n--- Epoch {epoch+1}/{TOTAL_EPOCHS} ---")

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # ✅ FIX #4: Gunakan AMP untuk mempercepat training
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

    # Validasi
    model.eval()
    val_correct = 0
    val_total   = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
            _, predicted = outputs.max(1)
            val_total   += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"✅ Validation Acc: {val_acc:.2f}%")

    # Simpan checkpoint
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc':             best_acc
    }, last_ckpt_path)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"🌟 Model terbaik baru! (Acc: {best_acc:.2f}%)")

# 6. EVALUASI AKHIR
print("\n" + "="*50)
print("⏳ MEMULAI UJI KLINIS MODEL...")
print("="*50)

# ✅ FIX #3: Tambah map_location=device
model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Mencetak Rapor"):
        images = images.to(device)
        with autocast():
            outputs = model(images)
        _, predicted = outputs.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

y_true_names = [kelas_penyakit[i] for i in y_true]
y_pred_names = [kelas_penyakit[i] for i in y_pred]

print("\n🏆 HASIL EVALUASI EFFICIENTNET-V2-S (640px) 🏆")
laporan_text = classification_report(y_true_names, y_pred_names,
                                     target_names=kelas_penyakit, digits=2)
print(laporan_text)

# 7. GRAFIK BAR
report_dict = classification_report(y_true_names, y_pred_names,
                                    target_names=kelas_penyakit, output_dict=True)
df_plot = pd.DataFrame([{"Kelas": k, "F1": v['f1-score']*100}
                         for k, v in report_dict.items()
                         if k in kelas_penyakit]).sort_values("F1")

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(df_plot['Kelas'], df_plot['F1'], color='#27ae60')
ax.set_xlabel('Tingkat Akurasi (F1-Score) dalam Persen %')
ax.set_title(f'Grafik Akurasi EfficientNetV2-S (Total: {round(report_dict["accuracy"]*100, 1)}%)')
ax.set_xlim(0, 105)
for bar in bars:
    width   = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    ax.text(width + 1, label_y, s=f'{width:.1f}%',
            va='center', fontweight='bold', color='#333')
plt.tight_layout()

grafik_path = f"{drive_path}/Grafik_Akurasi_EffV2S.png"
plt.savefig(grafik_path)
print(f"\n✅ Grafik tersimpan di: {grafik_path}")
plt.show()
