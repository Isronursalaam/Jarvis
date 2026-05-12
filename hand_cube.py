"""
Hand-Controlled 3D Cube - Virtual Reality Hand Tracking
=========================================================
Menggunakan MediaPipe Tasks API untuk deteksi tangan dan OpenCV untuk
rendering kubus 3D. Kubus bisa digerakkan menggunakan gerakan tangan.

Kontrol:
- 1 tangan: Pinch untuk pindahkan blok (lepas untuk pasang/snap ke grid)
- 2 tangan: Pinch keduanya untuk putar bangunan (seperti setir) & Zoom
- Telunjuk: Tunjuk untuk TAMBAH blok baru di posisi jari
- Genggam: Kepalkan tangan untuk gaya tarik BLACK HOLE!
- Buka Tangan: Tembakkan medan gaya REPULSOR! (Fisika)
- Peace (✌️): Nyalakan/Matikan GRAVITASI Bumi (Juggling Mode)
- Buka tangan untuk meletakkan blok yang dipegang
- Tekan 'R' untuk reset semua
- Tekan 'Q' atau ESC untuk keluar
- Tekan 'Q' atau ESC untuk keluar
"""

import cv2
import numpy as np
import math
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# mediapipe Tasks API

# ============================================================
# KONFIGURASI
# ============================================================
WINDOW_NAME = "Hand-Controlled 3D Cube"
MODEL_PATH = "hand_landmarker.task"
CUBE_SIZE = 80
GRAB_THRESHOLD = 0.05   # Jarak pinch (normalized, 0-1)
SMOOTHING = 0.3
MIN_SCALE = 0.3         # Skala minimum kubus
MAX_SCALE = 4.0         # Skala maksimum kubus
SCALE_SMOOTHING = 0.2   # Smoothing untuk zoom

# Warna (BGR)
FACE_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
]
EDGE_COLOR = (255, 255, 255)
GRAB_EDGE_COLOR = (0, 255, 0)
FACE_ALPHA = 0.4

# Kubus 3D
CUBE_VERTICES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64) * CUBE_SIZE / 2

CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

CUBE_FACES = [
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5],
]


def rotation_matrix(ax, ay, az):
    """Membuat rotation matrix dari sudut Euler (radian)."""
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project_3d(pts, cx, cy, fov=500):
    """Proyeksi perspektif 3D ke 2D."""
    result = []
    for p in pts:
        z = p[2] + fov
        if z <= 0:
            z = 1
        s = fov / z
        result.append((int(p[0] * s + cx), int(p[1] * s + cy)))
    return result


def draw_cube(frame, cube, is_grabbed_or_scaled, g_ax, g_ay, g_az, g_scale):
    """Menggambar objek kubus dengan state tertentu."""
    
    # === 1. Draw Energy Trails (Meteor Tail) ===
    if len(cube.trail) > 1:
        for i in range(1, len(cube.trail)):
            thickness = int((i / len(cube.trail)) * 10) + 2
            alpha = i / len(cube.trail)
            c_val = int(255 * alpha)
            cv2.line(frame, cube.trail[i-1], cube.trail[i], (c_val, c_val, 0), thickness, cv2.LINE_AA)

    # === 2. Draw Cube ===
    overlay = frame.copy()

    # Hitung transformasi vertex
    rot = rotation_matrix(g_ax, g_ay, g_az)
    scaled_verts = CUBE_VERTICES * g_scale
    rotated = (rot @ scaled_verts.T).T
    verts_2d = project_3d(rotated, cube.screen_x, cube.screen_y)
    verts_3d = rotated

    # Depth sort faces
    face_z = []
    for i, f in enumerate(CUBE_FACES):
        avg_z = np.mean([verts_3d[v][2] for v in f])
        face_z.append((avg_z, i))
    face_z.sort(key=lambda x: -x[0])

    # Draw faces
    for _, fi in face_z:
        pts = np.array([verts_2d[v] for v in CUBE_FACES[fi]], np.int32)
        cv2.fillConvexPoly(overlay, pts, cube.colors[fi])
    cv2.addWeighted(overlay, FACE_ALPHA, frame, 1 - FACE_ALPHA, 0, frame)

    # Draw edges
    ec = GRAB_EDGE_COLOR if is_grabbed_or_scaled else EDGE_COLOR
    th = 3 if is_grabbed_or_scaled else 2
    for v1, v2 in CUBE_EDGES:
        cv2.line(frame, verts_2d[v1], verts_2d[v2], ec, th, cv2.LINE_AA)
    for pt in verts_2d:
        cv2.circle(frame, pt, 3, ec, -1, cv2.LINE_AA)


def draw_ui(frame, fps, grabbed, is_scaling, total_cubes):
    """Menggambar overlay UI."""
    h, w = frame.shape[:2]
    # Header
    cv2.rectangle(frame, (0, 0), (w, 45), (30, 30, 30), -1)
    cv2.putText(frame, "Hand-Controlled 3D Cube", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f} | Cubes: {total_cubes}", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

    # Status
    if is_scaling:
        st = "ZOOM & ROTATE!"
        sc = (255, 200, 0)
    elif grabbed:
        st = "GRABBED!"
        sc = (0, 255, 0)
    else:
        st = "Open Hand"
        sc = (100, 100, 255)
    cv2.putText(frame, f"Status: {st}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2, cv2.LINE_AA)

    # Instructions
    instructions = [
        "Pinch: Pindah/Zoom | Telunjuk: Tambah | Genggam: PULL",
        "Terbuka: REPULSOR | Peace (V): GRAVITASI ON/OFF",
        "R: Reset | Q/ESC: Keluar",
    ]
    y0 = h - 85
    cv2.rectangle(frame, (0, y0 - 10), (w, h), (30, 30, 30), -1)
    for i, txt in enumerate(instructions):
        cv2.putText(frame, txt, (10, y0 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def draw_tony_stark_hand(frame, landmarks, w, h):
    """Gambar skeleton tangan dengan efek hologram khas Tony Stark (Iron Man)."""
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),     # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17),            # Palm
    ]

    points = []
    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Draw glowing lines (Efek Hologram)
    for c1, c2 in connections:
        if c1 < len(points) and c2 < len(points):
            # Glow luar (Cyan/Biru Muda Gelap)
            cv2.line(frame, points[c1], points[c2], (255, 150, 0), 6, cv2.LINE_AA)
            # Inti bercahaya (Putih kebiruan)
            cv2.line(frame, points[c1], points[c2], (255, 255, 200), 2, cv2.LINE_AA)

    # Draw hologram nodes (Sendi)
    for pt in points:
        cv2.circle(frame, pt, 4, (255, 255, 0), -1, cv2.LINE_AA) # Inti titik
        cv2.circle(frame, pt, 8, (255, 200, 0), 1, cv2.LINE_AA)  # Cincin luar


class Cube:
    def __init__(self, grid_x, grid_y, cx, cy):
        self.grid_x = grid_x
        self.grid_y = grid_y
        
        # Screen position (for smooth movement and rendering)
        self.screen_x = cx + grid_x * CUBE_SIZE
        self.screen_y = cy + grid_y * CUBE_SIZE
        self.is_grabbed = False
        
        # Variabel Fisika & Visual
        self.vx = 0.0
        self.vy = 0.0
        self.is_physics = False
        self.trail = []
        
        # Berikan warna acak untuk tiap kubus berdasarkan warna dasar
        self.colors = []
        for c in FACE_COLORS:
            b_shift = np.random.randint(-40, 40)
            g_shift = np.random.randint(-40, 40)
            r_shift = np.random.randint(-40, 40)
            self.colors.append((
                max(0, min(255, c[0] + b_shift)),
                max(0, min(255, c[1] + g_shift)),
                max(0, min(255, c[2] + r_shift))
            ))

    def update(self, cx, cy, g_scale, frame_w, frame_h, global_gravity):
        # Update Energy Trail
        if self.is_physics or self.is_grabbed:
            self.trail.append((int(self.screen_x), int(self.screen_y)))
            if len(self.trail) > 15:
                self.trail.pop(0)
        elif len(self.trail) > 0:
            self.trail.pop(0)

        if global_gravity and not self.is_grabbed:
            self.is_physics = True

        if self.is_physics:
            if global_gravity:
                self.vy += 0.8 # Gravitasi Bumi

            # Terapkan gaya kecepatan (Simulasi Fisika 2D)
            self.screen_x += self.vx
            self.screen_y += self.vy
            
            # Gesekan (Friction) udara agar melambat
            self.vx *= 0.95
            if not global_gravity:
                self.vy *= 0.95
            
            # Pantulan pinggir layar
            radius = (CUBE_SIZE * g_scale) / 2
            if self.screen_x < radius:
                self.screen_x = radius
                self.vx *= -0.8
            elif self.screen_x > frame_w - radius:
                self.screen_x = frame_w - radius
                self.vx *= -0.8
                
            if self.screen_y < radius:
                self.screen_y = radius
                self.vy *= -0.8
            elif self.screen_y > frame_h - radius:
                self.screen_y = frame_h - radius
                if self.vy > 2.0:
                    self.vy *= -0.7 # Bounce lantai
                else:
                    self.vy = 0 # Berhenti mantul
                self.vx *= 0.8 # Gesekan lantai
                
            # Berhenti dan lakukan "Snap" kembali ke Grid HANYA jika gravitasi mati
            if not global_gravity and abs(self.vx) < 1.0 and abs(self.vy) < 1.0:
                self.is_physics = False
                self.vx = 0
                self.vy = 0
                
                # Kalkulasi lokasi Grid baru dari layar
                cell_size = CUBE_SIZE * g_scale
                self.grid_x = round((self.screen_x - cx) / cell_size)
                self.grid_y = round((self.screen_y - cy) / cell_size)
                
        elif not self.is_grabbed:
            # Smoothly move to grid position
            target_sx = cx + self.grid_x * (CUBE_SIZE * g_scale)
            target_sy = cy + self.grid_y * (CUBE_SIZE * g_scale)
            self.screen_x += (target_sx - self.screen_x) * SMOOTHING
            self.screen_y += (target_sy - self.screen_y) * SMOOTHING

    def is_point_inside(self, px, py, g_scale):
        # Radius perkiraan berdasarkan scale
        radius = (CUBE_SIZE * g_scale) * 0.8
        dist = math.sqrt((self.screen_x - px)**2 + (self.screen_y - py)**2)
        return dist < radius


def get_empty_grid(cubes, start_x=0, start_y=0):
    """Mencari posisi grid kosong terdekat dari titik awal."""
    radius = 0
    while True:
        # Spiral search sederhana
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                test_x = start_x + dx
                test_y = start_y + dy
                occupied = any(c.grid_x == test_x and c.grid_y == test_y for c in cubes)
                if not occupied:
                    return test_x, test_y
        radius += 1


def is_pointing_gesture(hand_lms):
    """Mendeteksi apakah tangan sedang melakukan gesture 'menunjuk' (hanya telunjuk yang lurus)."""
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    wrist = hand_lms[0]
    # Index jari lurus (tip lebih jauh dari wrist dibanding pip)
    idx_ext = dist(wrist, hand_lms[8]) > dist(wrist, hand_lms[6])
    # Jari lain ditekuk (tip lebih dekat ke wrist dibanding pip)
    mid_curl = dist(wrist, hand_lms[12]) < dist(wrist, hand_lms[10])
    rng_curl = dist(wrist, hand_lms[16]) < dist(wrist, hand_lms[14])
    pnk_curl = dist(wrist, hand_lms[20]) < dist(wrist, hand_lms[18])
    
    return idx_ext and mid_curl and rng_curl and pnk_curl


def is_fist_gesture(hand_lms):
    """Mendeteksi gesture mengepal (semua 4 jari utama ditekuk ke dalam)."""
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    wrist = hand_lms[0]
    idx_curl = dist(wrist, hand_lms[8]) < dist(wrist, hand_lms[6])
    mid_curl = dist(wrist, hand_lms[12]) < dist(wrist, hand_lms[10])
    rng_curl = dist(wrist, hand_lms[16]) < dist(wrist, hand_lms[14])
    pnk_curl = dist(wrist, hand_lms[20]) < dist(wrist, hand_lms[18])
    
    return idx_curl and mid_curl and rng_curl and pnk_curl


def is_open_hand_gesture(hand_lms):
    """Mendeteksi telapak tangan terbuka penuh (Repulsor Mode)."""
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    wrist = hand_lms[0]
    idx_ext = dist(wrist, hand_lms[8]) > dist(wrist, hand_lms[6])
    mid_ext = dist(wrist, hand_lms[12]) > dist(wrist, hand_lms[10])
    rng_ext = dist(wrist, hand_lms[16]) > dist(wrist, hand_lms[14])
    pnk_ext = dist(wrist, hand_lms[20]) > dist(wrist, hand_lms[18])
    
    return idx_ext and mid_ext and rng_ext and pnk_ext


def is_peace_gesture(hand_lms):
    """Mendeteksi gaya tangan 'Peace' / V-Sign (Hanya telunjuk & tengah lurus)."""
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    wrist = hand_lms[0]
    idx_ext = dist(wrist, hand_lms[8]) > dist(wrist, hand_lms[6])
    mid_ext = dist(wrist, hand_lms[12]) > dist(wrist, hand_lms[10])
    rng_curl = dist(wrist, hand_lms[16]) < dist(wrist, hand_lms[14])
    pnk_curl = dist(wrist, hand_lms[20]) < dist(wrist, hand_lms[18])
    
    return idx_ext and mid_ext and rng_curl and pnk_curl


def main():
    print("=" * 50)
    print("  Hand-Controlled 3D Cube")
    print("=" * 50)
    print("Mencari kamera yang tersedia...")
    print("(Menggunakan DirectShow untuk IriUn Webcam)")

    # Coba DirectShow backend dulu (fix green screen IriUn)
    # Lalu fallback ke default jika gagal
    cap = None
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Default", cv2.CAP_ANY),
    ]

    for backend_name, backend in backends:
        for idx in range(5):
            test_cap = cv2.VideoCapture(idx, backend)
            if test_cap.isOpened():
                # Set MJPG codec (fix warna hijau IriUn)
                test_cap.set(cv2.CAP_PROP_FOURCC,
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                # Baca beberapa frame untuk stabilkan kamera
                good = False
                for _ in range(10):
                    ret, frame = test_cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # Cek apakah bukan green screen (rata-rata hijau terlalu tinggi)
                        mean_color = frame.mean(axis=(0, 1))
                        if mean_color[1] < 200 or abs(mean_color[0] - mean_color[1]) > 20:
                            good = True
                            break
                    time.sleep(0.05)

                if good:
                    print(f"  [OK] Kamera di index {idx} ({backend_name})")
                    cap = test_cap
                    break
                else:
                    # Tetap pakai kamera ini jika tidak ada yang lain
                    if cap is None:
                        cap = test_cap
                        print(f"  [?] Kamera di index {idx} ({backend_name}) - mungkin green")
                    else:
                        test_cap.release()
            else:
                test_cap.release()
        if cap is not None:
            # Cek jika sudah good, langsung keluar
            ret_test, f_test = cap.read()
            if ret_test and f_test is not None:
                m = f_test.mean(axis=(0, 1))
                if m[1] < 200 or abs(m[0] - m[1]) > 20:
                    break
            # Jika masih green, coba backend berikutnya
            if backend_name == "DirectShow":
                cap.release()
                cap = None

    if cap is None:
        print("ERROR: Tidak ada kamera yang ditemukan!")
        print("Pastikan IriUn webcam sudah aktif dan terhubung.")
        return

    # Baca frame untuk dapatkan resolusi aktual
    for _ in range(10):
        ret, test_frame = cap.read()
        if ret and test_frame is not None and test_frame.size > 0:
            break
        time.sleep(0.1)

    if not ret or test_frame is None:
        print("ERROR: Kamera terbuka tapi tidak bisa baca frame!")
        cap.release()
        return

    h, w = test_frame.shape[:2]
    print(f"Resolusi: {w}x{h}")

    # Variabel untuk menyimpan hasil deteksi dari callback
    detection_result_list = [None]
    timestamp_ms = [0]

    def result_callback(result, output_image, ts):
        detection_result_list[0] = result

    # Setup MediaPipe Hand Landmarker (LIVE_STREAM mode)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    # Center point layar (titik 0,0 dari grid dunia)
    cx, cy = w / 2, h / 2

    # State Aplikasi Global
    cubes = [Cube(0, 0, cx, cy)]
    active_cube = None
    
    # State Rotasi & Skala Dunia Global
    global_angle_x = -0.5
    global_angle_y = 0.5
    global_angle_z = 0.0
    global_scale = 1.0
    target_global_scale = 1.0
    
    is_grabbed = False
    is_scaling = False
    grab_offset_x, grab_offset_y = 0, 0
    
    # State untuk Zoom & Rotate 2 tangan
    initial_hand_dist = None
    initial_scale = 1.0
    initial_hand_angle = 0.0
    initial_angle_x, initial_angle_y, initial_angle_z = 0.0, 0.0, 0.0
    initial_center_x, initial_center_y = 0.0, 0.0
    
    last_spawn_time = 0.0
    
    # State Gravity Warp
    global_gravity = False
    last_gravity_time = 0.0

    prev_time = time.time()
    fps = 0
    frame_count = 0

    print("Siap! Gunakan tangan untuk mengontrol kubus.")

    # Buat window yang bisa di-resize dan di-fullscreen
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Kirim frame ke MediaPipe (async LIVE_STREAM)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms[0] = int(time.time() * 1000)

        try:
            landmarker.detect_async(mp_image, timestamp_ms[0])
        except Exception:
            pass  # Skip jika timestamp tidak valid

        # Proses hasil deteksi
        result = detection_result_list[0]
        hand_detected = False
        num_hands = 0
        hand_data = []  # Simpan data tiap tangan

        if result and result.hand_landmarks:
            num_hands = len(result.hand_landmarks)

            for hand_lms in result.hand_landmarks:
                hand_detected = True

                # Index finger tip (landmark 8), Thumb tip (landmark 4)
                idx_tip = hand_lms[8]
                thm_tip = hand_lms[4]

                ix, iy = idx_tip.x, idx_tip.y
                tx, ty = thm_tip.x, thm_tip.y

                # Pixel coords
                ipx, ipy = int(ix * w), int(iy * h)
                tpx, tpy = int(tx * w), int(ty * h)

                # Pinch distance (normalized)
                pinch = math.sqrt((ix - tx) ** 2 + (iy - ty) ** 2)

                # Mid point (pixel)
                mx = (ix * w + tx * w) / 2
                my = (iy * h + ty * h) / 2

                is_pinching = pinch < GRAB_THRESHOLD
                is_pointing = is_pointing_gesture(hand_lms)
                is_fist = is_fist_gesture(hand_lms)
                is_open = is_open_hand_gesture(hand_lms)
                is_peace = is_peace_gesture(hand_lms)

                # Simpan titik koordinat 2D untuk collision juggling
                lms_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

                hand_data.append({
                    'ix': ix, 'iy': iy, 'tx': tx, 'ty': ty,
                    'ipx': ipx, 'ipy': ipy, 'tpx': tpx, 'tpy': tpy,
                    'pinch': pinch, 'mx': mx, 'my': my,
                    'is_pinching': is_pinching,
                    'is_pointing': is_pointing,
                    'is_fist': is_fist,
                    'is_open': is_open,
                    'is_peace': is_peace,
                    'lms_px': lms_px
                })

                # Draw hologram hand skeleton
                draw_tony_stark_hand(frame, hand_lms, w, h)

            # === LOGIK MENCARI & BERINTERAKSI DENGAN KUBUS ===
            
            # 1. Cek gesture Pointing (Menunjuk) untuk menambah kubus
            any_pointing = next((hd for hd in hand_data if hd['is_pointing'] and not hd['is_pinching']), None)
            if any_pointing:
                # Visual indikator sedang menunjuk
                cv2.circle(frame, (any_pointing['ipx'], any_pointing['ipy']), 25, (255, 100, 255), 2, cv2.LINE_AA)
                
                # Cooldown 1 detik antar penambahan blok
                if time.time() - last_spawn_time > 1.0:
                    cell_size = CUBE_SIZE * global_scale
                    gx = round((any_pointing['ipx'] - cx) / cell_size)
                    gy = round((any_pointing['ipy'] - cy) / cell_size)
                    
                    if any(c.grid_x == gx and c.grid_y == gy for c in cubes):
                        gx, gy = get_empty_grid(cubes, start_x=gx, start_y=gy)
                        
                    cubes.append(Cube(gx, gy, cx, cy))
                    last_spawn_time = time.time()
                    print(f"Kubus ditambah via Pointing! Total: {len(cubes)}")

            # 2. Cek gesture Fist (Mengepal) untuk BLACK HOLE (Menarik semua kubus)
            any_fist = next((hd for hd in hand_data if hd['is_fist'] and not hd['is_pinching']), None)
            if any_fist:
                palm_x, palm_y = int(any_fist['mx']), int(any_fist['my'])
                
                # Visual Efek Black Hole (Pusaran Ungu/Hitam)
                t = time.time() * 20
                r1 = int(15 + 5 * math.sin(t))
                r2 = int(35 + 10 * math.sin(t*0.5))
                cv2.circle(frame, (palm_x, palm_y), r1, (255, 0, 200), -1, cv2.LINE_AA) # Inti ungu
                cv2.circle(frame, (palm_x, palm_y), r2, (100, 0, 150), 4, cv2.LINE_AA) # Cincin gelap
                cv2.putText(frame, "PULL!", (palm_x - 30, palm_y - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 200), 2, cv2.LINE_AA)
                
                # Force Physics (PULL / Tarik ke arah tangan)
                for c in cubes:
                    if c.is_grabbed: continue
                    dx = palm_x - c.screen_x
                    dy = palm_y - c.screen_y
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist > 30: # Batas agar tidak saling bertumpuk terlalu parah di tengah
                        c.vx += (dx / dist) * 3.5  # Kekuatan tarikan konstan
                        c.vy += (dy / dist) * 3.5
                        c.is_physics = True

            # 3. Cek gesture Open Hand (Repulsor Blast / Force Field)
            any_open = next((hd for hd in hand_data if hd['is_open'] and not hd['is_pinching'] and not hd['is_pointing']), None)
            if any_open:
                palm_x, palm_y = int(any_open['mx']), int(any_open['my'])
                
                # Visual Efek Repulsor (Tony Stark Style)
                t = time.time() * 15
                r1 = int(25 + 5 * math.sin(t))
                r2 = int(45 + 10 * math.sin(t*0.8))
                r3 = int(60 + 15 * math.sin(t*0.5))
                cv2.circle(frame, (palm_x, palm_y), r1, (255, 255, 200), -1, cv2.LINE_AA) # Inti terang
                cv2.circle(frame, (palm_x, palm_y), r2, (255, 255, 0), 4, cv2.LINE_AA) # Cincin Cyan
                cv2.circle(frame, (palm_x, palm_y), r3, (200, 200, 0), 2, cv2.LINE_AA) # Cincin luar
                
                # Force Physics ke semua kubus (tolak/repel)
                for c in cubes:
                    if c.is_grabbed: continue
                    dx = c.screen_x - palm_x
                    dy = c.screen_y - palm_y
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < 600: # Jangkauan Repulsor
                        force = (600 - dist) * 0.15 # Semakin dekat semakin kuat
                        if dist == 0: dist = 1
                        c.vx += (dx / dist) * force
                        c.vy += (dy / dist) * force
                        c.is_physics = True

            # 4. Cek gesture Peace (Toggle GRAVITY & JUGGLING MODE)
            any_peace = next((hd for hd in hand_data if hd['is_peace']), None)
            if any_peace and time.time() - last_gravity_time > 1.0:
                global_gravity = not global_gravity
                last_gravity_time = time.time()
                status_txt = "GRAVITASI ON" if global_gravity else "GRAVITASI OFF"
                print(f"[{status_txt}]")

            # === FISIKA TANGAN (JUGGLING MODE) ===
            # Jadikan seluruh 21 titik sendi tangan sebagai benda padat (bisa memantulkan balok)
            if global_gravity or any(c.is_physics for c in cubes):
                for hd in hand_data:
                    for px, py in hd['lms_px']:
                        for c in cubes:
                            if c.is_grabbed: continue
                            dx = c.screen_x - px
                            dy = c.screen_y - py
                            dist = math.sqrt(dx**2 + dy**2)
                            min_dist = (CUBE_SIZE * global_scale) * 0.5 + 15 # Radius kubus + radius node jari
                            if dist < min_dist:
                                if dist == 0: dist = 1
                                # Penetration resolution (Dorong keluar)
                                overlap = min_dist - dist
                                nx = dx / dist
                                ny = dy / dist
                                c.screen_x += nx * overlap
                                c.screen_y += ny * overlap
                                
                                # Memantulkan kecepatan (Bouncing)
                                dot = c.vx * nx + c.vy * ny
                                if dot < 0: # Hanya memantul jika bergerak saling mendekat
                                    c.vx -= 1.8 * dot * nx
                                    c.vy -= 1.8 * dot * ny
                                    # Tambahan tenaga pukulan dari tangan (Bump up)
                                    c.vy -= 3.0
                                    c.is_physics = True

            # Cek apakah ada tangan yang pinching
            any_pinching = any(hd['is_pinching'] for hd in hand_data)
            
            if not any_pinching:
                # Kubus baru saja dilepas! Lakukan SNAPPING
                if is_grabbed and active_cube is not None:
                    # Cari sel grid terdekat dari posisi layar saat ini
                    cell_size = CUBE_SIZE * global_scale
                    gx = round((active_cube.screen_x - cx) / cell_size)
                    gy = round((active_cube.screen_y - cy) / cell_size)
                    
                    # Jika tertumpuk dengan blok lain, cari posisi kosong terdekat
                    if any(c.grid_x == gx and c.grid_y == gy and c != active_cube for c in cubes):
                        gx, gy = get_empty_grid(cubes, start_x=gx, start_y=gy)
                        
                    active_cube.grid_x = gx
                    active_cube.grid_y = gy
                    active_cube.is_grabbed = False
                
                active_cube = None
                is_grabbed = False
                is_scaling = False
                initial_hand_dist = None

            # Mode 2 Tangan: Zoom & Rotate Global
            elif num_hands == 2 and hand_data[0]['is_pinching'] and hand_data[1]['is_pinching']:
                # Hentikan grab jika sedang pindah blok
                if is_grabbed and active_cube is not None:
                    active_cube.is_grabbed = False
                active_cube = None
                is_scaling = True
                is_grabbed = False

                dx = hand_data[1]['ipx'] - hand_data[0]['ipx']
                dy = hand_data[1]['ipy'] - hand_data[0]['ipy']
                d = math.sqrt(dx**2 + dy**2)
                current_angle = math.atan2(dy, dx)
                
                zmx = (hand_data[0]['ipx'] + hand_data[1]['ipx']) / 2
                zmy = (hand_data[0]['ipy'] + hand_data[1]['ipy']) / 2

                if initial_hand_dist is None:
                    initial_hand_dist = d
                    initial_scale = global_scale
                    initial_hand_angle = current_angle
                    initial_angle_x = global_angle_x
                    initial_angle_y = global_angle_y
                    initial_angle_z = global_angle_z
                    initial_center_x = zmx
                    initial_center_y = zmy
                else:
                    if initial_hand_dist > 0:
                        ratio = d / initial_hand_dist
                        target_global_scale = np.clip(initial_scale * ratio, MIN_SCALE, MAX_SCALE)
                    
                    # Rotasi global
                    global_angle_z = initial_angle_z + (current_angle - initial_hand_angle)
                    dx_center = zmx - initial_center_x
                    dy_center = zmy - initial_center_y
                    global_angle_y = initial_angle_y + (dx_center * 0.01)
                    global_angle_x = initial_angle_x + (dy_center * 0.01)

                # Gambar UI Zoom
                cv2.line(frame,
                         (hand_data[0]['ipx'], hand_data[0]['ipy']),
                         (hand_data[1]['ipx'], hand_data[1]['ipy']),
                         (255, 200, 0), 2, cv2.LINE_AA)
                radius = int(d / 4)
                cv2.circle(frame, (int(zmx), int(zmy)), radius, (255, 200, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"{global_scale:.1f}x", (int(zmx) - 20, int(zmy) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)

            # Mode 1 Tangan: Grab & Move Individual Block
            elif num_hands >= 1:
                is_scaling = False
                initial_hand_dist = None
                
                hd = next((h for h in hand_data if h['is_pinching']), None)
                
                if hd:
                    if active_cube is None:
                        for c in reversed(cubes):
                            if c.is_point_inside(hd['mx'], hd['my'], global_scale):
                                active_cube = c
                                c.is_grabbed = True
                                grab_offset_x = c.screen_x - hd['mx']
                                grab_offset_y = c.screen_y - hd['my']
                                is_grabbed = True
                                break
                    
                    if active_cube:
                        is_grabbed = True
                        active_cube.screen_x = hd['mx'] + grab_offset_x
                        active_cube.screen_y = hd['my'] + grab_offset_y

            # Draw pinch indicators
            for hd in hand_data:
                color = (0, 255, 0) if hd['is_pinching'] else (0, 200, 255)
                cv2.line(frame, (hd['tpx'], hd['tpy']), (hd['ipx'], hd['ipy']), color, 2, cv2.LINE_AA)
                cv2.circle(frame, (hd['ipx'], hd['ipy']), 10, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (hd['tpx'], hd['tpy']), 10, (255, 200, 0), 2, cv2.LINE_AA)
                if hd['is_pinching']:
                    cv2.circle(frame, (int(hd['mx']), int(hd['my'])), 15, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.circle(frame, (int(hd['mx']), int(hd['my'])), 5, (0, 255, 0), -1, cv2.LINE_AA)

        else:
            # Tidak ada tangan yang terdeteksi
            if is_grabbed and active_cube is not None:
                # Snap juga jika tiba-tiba tangan keluar layar saat menggeser
                cell_size = CUBE_SIZE * global_scale
                gx = round((active_cube.screen_x - cx) / cell_size)
                gy = round((active_cube.screen_y - cy) / cell_size)
                if any(c.grid_x == gx and c.grid_y == gy and c != active_cube for c in cubes):
                    gx, gy = get_empty_grid(cubes, start_x=gx, start_y=gy)
                active_cube.grid_x = gx
                active_cube.grid_y = gy
                active_cube.is_grabbed = False
                
            active_cube = None
            is_grabbed = False
            is_scaling = False
            initial_hand_dist = None

        # Smooth global scale
        global_scale += (target_global_scale - global_scale) * SCALE_SMOOTHING
        
        # Indikator Layar Gravitasi
        if global_gravity:
            cv2.putText(frame, "GRAVITY: ON (JUGGLING MODE)", (w//2 - 150, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Update dan Render semua kubus
        for c in cubes:
            c.update(cx, cy, global_scale, w, h, global_gravity)
            is_interacting = (c == active_cube) and is_grabbed
            draw_cube(frame, c, is_interacting, global_angle_x, global_angle_y, global_angle_z, global_scale)

        # FPS
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_time = now

        draw_ui(frame, fps, is_grabbed, is_scaling, len(cubes))
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            # Reset ke 1 kubus di tengah
            cubes = [Cube(0, 0, cx, cy)]
            active_cube = None
            global_angle_x = -0.5
            global_angle_y = 0.5
            global_angle_z = 0.0
            global_scale = 1.0
            target_global_scale = 1.0
            is_grabbed = False
            is_scaling = False
            initial_hand_dist = None
            print("Reset! Kembali ke 1 kubus.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Selesai.")


if __name__ == "__main__":
    main()
