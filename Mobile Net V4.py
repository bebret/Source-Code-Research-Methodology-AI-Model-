# ====================================================================
# 1. PASANG SABUK PENGAMAN (SAMBUNGKAN KE GOOGLE DRIVE)
# ====================================================================
from google.colab import drive
import os

drive.mount('/content/drive')

drive_path = "/content/drive/MyDrive/Skripsi_MobileNetV4"
os.makedirs(drive_path, exist_ok=True)

# ====================================================================
# 2. INSTALL LIBRARY (UPGRADE TIMM WAJIB UNTUK V4) & ROBOFLOW
# ====================================================================
!pip install -q roboflow scikit-learn matplotlib pandas
!pip install -q --upgrade timm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from roboflow import Roboflow
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Download Dataset format "Folder Classification"
rf = Roboflow(api_key="CU04tyVnI9U4FqyF9aPv")
project = rf.workspace("alberts-workspace-lrb7z").project("skin-disease-3-jjow2")
version = project.version(3)
dataset = version.download("folder")

# ====================================================================
# 3. PERSIAPAN DATASET (DIET AUGMENTASI)
# ====================================================================
print("\n⚙️ Menyiapkan Dataset & Augmentasi...")
# MobileNet umumnya optimal di resolusi 224x224 atau 256x256 (Biar RAM aman)
IMG_SIZE = 224
BATCH_SIZE = 16

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

train_dir = f"{dataset.location}/train"
val_dir = f"{dataset.location}/valid"

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

kelas_penyakit = train_data.classes
num_classes = len(kelas_penyakit)

# ====================================================================
# 4. INISIALISASI MOBILENET-V4 & AUTO-RESUME
# ====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Menggunakan Device: {device}")

# Memanggil MobileNetV4 versi Medium (Paling seimbang untuk akurasi & speed)
model = timm.create_model('mobilenetv4_conv_medium', pretrained=True, num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

last_ckpt_path = f"{drive_path}/last_mobilenetv4.pth"
best_ckpt_path = f"{drive_path}/best_mobilenetv4.pth"

start_epoch = 0
best_acc = 0.0

# LOGIKA AUTO-RESUME
if os.path.exists(last_ckpt_path):
    print("\n🔄 DITEMUKAN FILE LAST CHECKPOINT DI DRIVE!")
    print("🚀 Melanjutkan training yang sempat terputus...")
    checkpoint = torch.load(last_ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
else:
    print("\n🚀 MEMULAI TRAINING BARU DARI AWAL (EPOCH 1)...")

# ====================================================================
# 5. LOOPING TRAINING ("MESIN MANUAL")
# ====================================================================
TOTAL_EPOCHS = 35

for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\n--- Epoch {epoch+1}/{TOTAL_EPOCHS} ---")

    # Proses Belajar
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

    # Validasi (Ujian)
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"✅ Validation Acc: {val_acc:.2f}%")

    # --- SIMPAN CHECKPOINT KE DRIVE (SABUK PENGAMAN) ---
    # Simpan file 'last' setiap selesai 1 epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, last_ckpt_path)

    # Simpan file 'best' kalau akurasinya memecahkan rekor
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"🌟 Model terbaik baru tersimpan ke Drive! (Acc: {best_acc:.2f}%)")

# ====================================================================
# 6. EVALUASI AKHIR (MENGHITUNG PRECISION, RECALL, F1, SUPPORT)
# ====================================================================
print("\n" + "="*50)
print("⏳ MEMULAI UJI KLINIS MODEL UNTUK LAPORAN SKRIPSI...")
print("="*50)

# Load model terbaik yang ada di Drive untuk dievaluasi
model.load_state_dict(torch.load(best_ckpt_path))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\n🏆 HASIL EVALUASI AKHIR MODEL (MOBILENET-V4) 🏆")
laporan_text = classification_report(y_true, y_pred, target_names=kelas_penyakit, digits=2)
print(laporan_text)

# ====================================================================
# 7. BUAT GRAFIK BAR DAN SIMPAN KE DRIVE
# ====================================================================
report_dict = classification_report(y_true, y_pred, target_names=kelas_penyakit, output_dict=True)
f1_scores = {kelas: report_dict[kelas]['f1-score'] for kelas in kelas_penyakit}

df_plot = pd.DataFrame(list(f1_scores.items()), columns=['Kelas', 'F1-Score']).sort_values('F1-Score', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(df_plot['Kelas'], df_plot['F1-Score'] * 100, color='#ff7f0e') # Warna Oranye beda dari YOLO

ax.set_xlabel('Tingkat Akurasi (F1-Score) dalam Persen %')
ax.set_title(f'Grafik Akurasi MobileNet-V4 per Kelas (Total: {round(report_dict["accuracy"]*100, 1)}%)')
ax.set_xlim(0, 105)

for bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    ax.text(width + 1, label_y, s=f'{width:.1f}%', va='center', fontweight='bold', color='#333')

plt.tight_layout()

grafik_path = f"{drive_path}/Grafik_Akurasi_MobileNetV4.png"
plt.savefig(grafik_path)

print(f"\n✅ Grafik berhasil dibuat dan DISIMPAN PERMANEN di Google Drive:")
print(f"👉 Lokasi: {grafik_path}")

plt.show()
