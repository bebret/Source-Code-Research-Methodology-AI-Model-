# ====================================================================
# 1. PASANG SABUK PENGAMAN (SAMBUNGKAN KE GOOGLE DRIVE)
# ====================================================================
from google.colab import drive
import os

# Meminta izin akses ke Google Drive kamu
drive.mount('/content/drive')

# Membuat folder khusus di Drive kamu untuk menyimpan hasil skripsi ini
drive_path = "/content/drive/MyDrive/Skripsi_YOLO_Skin"
os.makedirs(drive_path, exist_ok=True)

# ====================================================================
# 2. INSTALL LIBRARY & DOWNLOAD DATASET ROBOFLOW
# ====================================================================
!pip install -q roboflow ultralytics scikit-learn matplotlib pandas

from roboflow import Roboflow
rf = Roboflow(api_key="CU04tyVnI9U4FqyF9aPv")
project = rf.workspace("alberts-workspace-lrb7z").project("skin-disease-3-jjow2")
version = project.version(3)
# Dataset akan terdownload secara lokal di mesin Colab
dataset = version.download("folder")

# ====================================================================
# 3. TRAINING DENGAN LOGIKA AUTO-RESUME & SIMPAN KE DRIVE
# ====================================================================
from ultralytics import YOLO

# Kita cek apakah di dalam Drive kamu sudah ada file last.pt (training sebelumnya)
last_weight_path = f"{drive_path}/train/weights/last.pt"

if os.path.exists(last_weight_path):
    print("\n🔄 DITEMUKAN FILE LAST.PT DI DRIVE!")
    print("🚀 Melanjutkan training yang sempat terputus (Auto-Resume)...")
    # Panggil model dari titik terakhir dia mati
    model = YOLO(last_weight_path)
    # Resume training
    model.train(resume=True)
else:
    print("\n🚀 MEMULAI TRAINING BARU DARI AWAL (EPOCH 1)...")
    model = YOLO('yolov8l-cls.pt') # Pakai arsitektur Large

    model.train(
        data=dataset.location,
        project=drive_path,    # ⚠️ PENTING: Semua hasil akan langsung disimpan ke Drive!
        epochs=35,
        imgsz=640,
        batch=8, # Turunkan agar RAM Colab tidak jebol

        # --- Diet Augmentasi (Biar Cepat) ---
        fliplr=0.5,
        degrees=10.0,
        mosaic=0.0,
        mixup=0.0,
        workers=2,
        cache=True
    )

# ====================================================================
# 4. EVALUASI AKHIR (MENGHITUNG PRECISION, RECALL, F1, SUPPORT)
# ====================================================================
print("\n" + "="*50)
print("⏳ MEMULAI UJI KLINIS MODEL UNTUK LAPORAN SKRIPSI...")
print("="*50)

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Ambil model TERBAIK dari hasil training yang tersimpan di Drive
best_weight_path = f"{drive_path}/train/weights/best.pt"
model_eval = YOLO(best_weight_path)

# Arahkan ke folder validasi dari Roboflow
val_dir = f"{dataset.location}/valid"
kelas_penyakit = sorted(os.listdir(val_dir))

y_true = []
y_pred = []

# AI akan menebak satu per satu gambar di folder validasi
for nama_kelas in kelas_penyakit:
    folder_kelas = os.path.join(val_dir, nama_kelas)
    if not os.path.isdir(folder_kelas): continue

    for file_gambar in os.listdir(folder_kelas):
        path_gambar = os.path.join(folder_kelas, file_gambar)

        # Suruh YOLO nebak (pakai trik anti-list yang sudah kita bahas)
        hasil = model_eval(path_gambar, verbose=False)
        prediksi = hasil
        while isinstance(prediksi, list): prediksi = prediksi

        tebakan_idx = prediksi.probs.top1
        nama_tebakan = prediksi.names[tebakan_idx]

        # Kumpulkan kunci jawaban (y_true) dan tebakan AI (y_pred)
        y_true.append(nama_kelas)
        y_pred.append(nama_tebakan)

# Cetak Laporan ala sklearn (Mirip screenshot-mu)
print("\n🏆 HASIL EVALUASI AKHIR MODEL 🏆")
laporan_text = classification_report(y_true, y_pred, target_names=kelas_penyakit, digits=2)
print(laporan_text)

# ====================================================================
# 5. BUAT GRAFIK BAR DAN SIMPAN KE DRIVE
# ====================================================================
# Tarik data F1-Score dari sklearn
report_dict = classification_report(y_true, y_pred, target_names=kelas_penyakit, output_dict=True)
f1_scores = {kelas: report_dict[kelas]['f1-score'] for kelas in kelas_penyakit}

# Urutkan dataframe
df_plot = pd.DataFrame(list(f1_scores.items()), columns=['Kelas', 'F1-Score']).sort_values('F1-Score', ascending=True)

# Gambar Grafiknya
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(df_plot['Kelas'], df_plot['F1-Score'] * 100, color='#2F80ED')

# Aksesoris Grafik
ax.set_xlabel('Tingkat Akurasi (F1-Score) dalam Persen %')
ax.set_title(f'Grafik Akurasi Model per Kelas Penyakit (Total Keseluruhan: {round(report_dict["accuracy"]*100, 1)}%)')
ax.set_xlim(0, 105)

# Tampilkan angka di sebelah bar
for bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    ax.text(width + 1, label_y, s=f'{width:.1f}%', va='center', fontweight='bold', color='#333')

plt.tight_layout()

# Simpan foto grafiknya ke Google Drive
grafik_path = f"{drive_path}/Grafik_Akurasi_Skripsi.png"
plt.savefig(grafik_path)

print(f"\n✅ Grafik berhasil dibuat dan DISIMPAN PERMANEN di Google Drive:")
print(f"👉 Lokasi: {grafik_path}")

plt.show()
