Tentu, ini adalah draf `README.md` yang keren, profesional, dan bergaya futuristik (ala Stark Industries) untuk proyek GitHub kamu.

---

# 🖐️ Hand-Controlled 3D Cube: Virtual Reality Hand Tracking

Ubah tanganmu menjadi kontroler VR! Proyek ini menggunakan **MediaPipe Tasks API** untuk deteksi tangan secara *real-time* dan **OpenCV** untuk melakukan rendering objek 3D tanpa memerlukan engine berat seperti Unity atau Unreal.

## 🚀 Fitur Utama

* **Holographic Hand Skeleton:** Visualisasi tangan bergaya Tony Stark (Iron Man).
* **Advanced Physics Engine:** Simulasi gravitasi, pantulan, gesekan, dan gaya tarik/dorong.
* **Gesture Recognition:** Berbagai gerakan tangan untuk memicu aksi yang berbeda.
* **Global World Control:** Rotasi dan zoom seluruh area kerja menggunakan dua tangan.
* **Grid Snapping:** Blok akan otomatis menempel pada grid saat dilepaskan.

---

## 🎮 Kontrol & Gestur

Kendalikan dunia virtualmu dengan gerakan tangan yang intuitif:

| Gestur | Aksi | Deskripsi |
| --- | --- | --- |
| **Pinch (Satu Tangan)** | 🤏 **Grab & Move** | Cubit untuk memindahkan blok individu ke posisi baru. |
| **Pinch (Dua Tangan)** | 👐 **Zoom & Rotate** | Seperti memutar setir; mendekat/menjauhkan tangan untuk zoom. |
| **Point (Telunjuk)** | ☝️ **Spawn** | Tunjuk area kosong untuk menambah blok baru secara instan. |
| **Fist (Genggam)** | ✊ **Black Hole** | Menciptakan medan gravitasi yang menarik semua blok ke tangan. |
| **Open Hand** | ✋ **Repulsor** | Menembakkan medan gaya (Force Field) untuk mendorong semua blok. |
| **Peace Sign (✌️)** | ✌️ **Gravity Warp** | Mengaktifkan/mematikan gravitasi bumi (Mode Juggling). |

### Tombol Keyboard:

* `R`: Reset seluruh simulasi ke posisi awal.
* `Q` / `ESC`: Keluar dari aplikasi.

---

## 🛠️ Persiapan Lingkungan

### Prasyarat

* Webcam (Bisa menggunakan Iriun Webcam untuk kualitas lebih baik).
* File Model: Unduh `hand_landmarker.task` dari MediaPipe dan letakkan di direktori yang sama dengan script.

### Instalasi Dependensi

```bash
pip install opencv-python mediapipe numpy

```

---

## 💡 Cara Kerja

Proyek ini memadukan matematika proyeksi 3D dengan Computer Vision:

1. **Detection:** MediaPipe menangkap 21 titik koordinat (*landmarks*) dari tangan manusia.
2. **Projection:** Mengonversi koordinat 3D dari kubus ke layar 2D menggunakan matriks rotasi dan formula perspektif:

$$s = \frac{fov}{z + fov}$$


3. **Physics:** Menghitung vektor kecepatan ($v_x, v_y$) saat terjadi interaksi gaya seperti *Repulsor* atau *Black Hole*.

---

## 📝 Catatan Teknis

* **DirectShow Support:** Script telah dioptimalkan untuk menangani masalah "Green Screen" pada driver webcam tertentu (seperti Iriun).
* **Smoothing:** Menggunakan teknik interpolasi linear agar pergerakan kubus tidak patah-patah meskipun ada fluktuasi pada deteksi tangan.

---

**Dibuat dengan ❤️ oleh [Nama Kamu]**
*Mari berinteraksi dengan dunia digital secara lebih alami!*
