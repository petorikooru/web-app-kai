import asyncio
import base64
import csv
import io
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import time
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("RailDetection")


# ============================================================================
# KONSTANTA GLOBAL (dibagikan ke semua proses)
# ============================================================================

CLASS_NAMES = {
    0: "Background",
    1: "Corrugations",
    2: "rail-end",
    3: "Rail Line",
    4: "Shelling",
    5: "Squat",
    6: "Wheelburns",
    7: "Wheelslip",
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 0),
    5: (0, 165, 255),
    6: (128, 0, 128),
    7: (0, 255, 255),
}

SKIP_CLASSES = {0, 3, 6, 7}


# ============================================================================
# CLASS 1 — WebBroadcaster
# Menangani pengiriman data (frame + log) dari proses ke browser via WebSocket
# ============================================================================


class WebBroadcaster:
    """
    Jembatan antara pipeline multiprocessing dan klien browser.

    Cara kerja:
      - BatchWorkerProcess menaruh hasil ke `broadcast_queue` (multiprocessing.Queue)
      - WebBroadcaster._poll_queue() dipanggil via asyncio loop di dalam WebServer
      - Setiap WebSocket klien yang connect didaftarkan ke `_clients`
      - Data dikirim ke semua klien secara async (broadcast)

    Format pesan WebSocket (JSON):
      {
        "type": "frame",           # atau "log" / "status" / "graph_start"
        "camera_id": 1,
        "frame_b64": "<base64>",   # JPEG frame
        "total_contours": 3,
        "gps": "...",
        "detected_classes": [{"name": "Squat", "count": 2}],
        "timestamp": "2024-..."
      }
    """

    def __init__(self, broadcast_queue: Queue):
        self.broadcast_queue = broadcast_queue
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def register(self, ws: WebSocket):
        async with self._lock:
            self._clients.append(ws)
        log.info(f"WebSocket klien terhubung. Total: {len(self._clients)}")

    async def unregister(self, ws: WebSocket):
        async with self._lock:
            self._clients = [c for c in self._clients if c is not ws]
        log.info(f"WebSocket klien terputus. Sisa: {len(self._clients)}")

    async def _send_to_all(self, message: dict):
        """Kirim pesan JSON ke semua klien yang terhubung."""
        if not self._clients:
            return
        text = json.dumps(message)
        disconnected = []
        async with self._lock:
            for ws in self._clients:
                try:
                    await ws.send_text(text)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                self._clients.remove(ws)

    async def poll_and_broadcast(self):
        """
        Loop async yang terus membaca broadcast_queue dan meneruskan ke browser.
        Dipanggil sebagai asyncio Task saat server start.
        """
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Non-blocking get agar tidak memblokir event loop
                msg = await loop.run_in_executor(
                    None, lambda: self.broadcast_queue.get(timeout=0.05)
                )
                await self._send_to_all(msg)
            except Exception:
                # Queue kosong atau timeout — tidak apa-apa
                await asyncio.sleep(0.01)


# ============================================================================
# CLASS 2 — WebServer
# FastAPI app + endpoint WebSocket + halaman HTML dashboard
# ============================================================================


class WebServer:
    """
    FastAPI server yang menjadi antarmuka web untuk pipeline.

    Endpoint:
      GET  /          → Dashboard HTML (embedded, tidak perlu file statis)
      WS   /ws        → Stream frame + log JSON ke browser
      POST /control   → Terima perintah start/stop/pause dari browser
    """

    def __init__(
        self,
        broadcaster: "WebBroadcaster",
        control_queue: Queue,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.broadcaster = broadcaster
        self.control_queue = control_queue
        self.host = host
        self.port = port
        self.app = FastAPI(title="Rail Crack Detection")
        self._register_routes()

    def _register_routes(self):
        broadcaster = self.broadcaster

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # startup
            task = asyncio.create_task(broadcaster.poll_and_broadcast())
            log.info("Broadcast loop dimulai.")
            yield
            # shutdown
            task.cancel()

        # Re-create app dengan lifespan agar tidak pakai on_event
        self.app = FastAPI(title="Rail Crack Detection", lifespan=lifespan)
        app = self.app

        app.mount("/icon_100_", StaticFiles(directory="icon_100_"), name="icon_100_")

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            # return _build_dashboard_html()
            return FileResponse("index.html")

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            await self.broadcaster.register(ws)
            try:
                # Tetap terbuka; klien bisa kirim perintah via WS juga
                while True:
                    data = await ws.receive_text()
                    try:
                        cmd = json.loads(data)

                        # --- TAMBAHAN UNTUK THRESHOLD & LOGGING ---
                        # action = cmd.get("action")
                        # if action == "set_threshold":
                        #     # Hanya print/log, lalu biarkan masuk ke queue
                        #     print(
                        #         f"[WEBSOCKET] Menerima perubahan threshold: {cmd.get('value')}"
                        #     )
                        # elif action == "set_mode":
                        #     print(
                        #         f"[WEBSOCKET] Menerima perubahan mode: {cmd.get('mode')}"
                        #     )
                        # ------------------------------------------

                        self.control_queue.put_nowait(cmd)
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                pass
            finally:
                await self.broadcaster.unregister(ws)

        @app.post("/control")
        async def control(payload: dict):
            """
            Terima perintah kontrol dari browser (HTTP POST).
            Payload contoh: {"action": "stop"} / {"action": "pause"} / {"action": "start"}
            """
            self.control_queue.put_nowait(payload)
            return {"status": "ok", "received": payload}

    def run(self):
        """Jalankan server di main process (blocking)."""
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")

    def run_in_process(self) -> Process:
        """Jalankan server di proses terpisah (non-blocking untuk main)."""
        p = Process(target=self.run, daemon=True, name="WebServerProcess")
        p.start()
        return p


# ============================================================================
# CLASS 3 — ImageSaverProcess
# Menggantikan ImageSaverWorker(QThread) → multiprocessing.Process
# ============================================================================
def _is_jpg(val: str) -> bool:
    return isinstance(val, str) and ".jpg" in val.lower()


def _is_defect_type(val: str) -> bool:
    """True jika val adalah nama jenis cacat (bukan jpg, dash, kosong, atau angka)."""
    if not val or val.strip() in ("", "-"):
        return False
    if _is_jpg(val):
        return False
    try:
        float(val)
        return False
    except ValueError:
        return True


def _safe(val) -> str:
    """Kembalikan '-' jika nilai None atau string kosong."""
    if val is None:
        return "-"
    v = str(val).strip()
    return v if v else "-"


def _parse_datetime(time_str: str):
    """'20260408_200305' → ('2026-04-08', '20:03:05')"""
    s = str(time_str).strip()
    try:
        dp = s[:8]
        tp = s[9:15]
        return f"{dp[:4]}-{dp[4:6]}-{dp[6:8]}", f"{tp[:2]}:{tp[2:4]}:{tp[4:6]}"
    except Exception:
        return "-", s


def _parse_gps(gps_str: str):
    """Ekstrak (lat, lon) dari string GPS. Return ('-', '-') jika belum ada sinyal."""
    s = str(gps_str).strip()
    if not s or s == "nan" or "Menunggu" in s or "waiting" in s.lower():
        return "-", "-"
    nums = re.findall(r"[-+]?\d+\.\d+", s)
    if len(nums) >= 2:
        return nums[0], nums[1]
    return s, "-"


def _classify_and_fix_row(row: list) -> dict:
    """
    Perbaiki satu raw CSV row yang kolom-kolomnya salah posisi.

    Mapping kolom CSV (header lama) → data aktual:
      [2] 'Accuracy'          → Left Defect Type  (Squat / Corrugations / -)
      [3] 'Loss'              → Left Total count   (0, 1, 2, ...)
      [4] 'Confidence'        → Right Total count  (0, 1, 2, ...)
      [5] 'Left Defect Type'  → Condition          (Kiri/Kanan Perlu Perbaikan)
      [6] 'Left Total'        → Confidence score   (0.9500)
      [7..9]                  → posisi gambar & defect kanan bervariasi per baris

    5 pola posisi col[7], col[8], col[9]:
      A : left only,  gambar di col[7]                → col7=cam1.jpg
      B : left only,  gambar di col[9] (2 kosong dulu)→ col9=cam1.jpg
      C : right only, defect di col[7], gambar col[8] → col7=Squat, col8=cam2.jpg
      D : right only, kosong col[7], defect+img col[8,9]
      E1: both cams,  left_img col[7], right_defect+img col[8,9]
      E2: both cams,  right_defect+img col[7,8], left_img col[9]
    """
    r = row + [""] * 5

    left_defect = r[2].strip() if _is_defect_type(r[2].strip()) else "-"
    right_defect = "-"
    left_image = "-"
    right_image = "-"

    col7 = r[7].strip()
    col8 = r[8].strip()
    col9 = r[9].strip()

    if _is_jpg(col7) and not col8 and not col9:  # A
        left_image = col7
    elif not col7 and not col8 and _is_jpg(col9):  # B
        left_image = col9
    elif _is_defect_type(col7) and _is_jpg(col8) and not col9:  # C
        right_defect, right_image, left_defect = col7, col8, "-"
    elif not col7 and _is_defect_type(col8) and _is_jpg(col9):  # D
        right_defect, right_image, left_defect = col8, col9, "-"
    elif _is_jpg(col7) and _is_defect_type(col8) and _is_jpg(col9):  # E1
        left_image, right_defect, right_image = col7, col8, col9
    elif _is_defect_type(col7) and _is_jpg(col8) and _is_jpg(col9):  # E2
        right_defect, right_image, left_image = col7, col8, col9

    return {
        "time_raw": r[0],
        "gps_raw": r[1],
        "left_defect": left_defect,
        "right_defect": right_defect,
        "left_total": _safe(r[3]),
        "right_total": _safe(r[4]),
        "conf_score": _safe(r[6]),
        "condition": _safe(r[5]),
        "left_image": _safe(left_image),
        "right_image": _safe(right_image),
    }


def _build_output_df(raw_rows: list) -> pd.DataFrame:
    """
    Ubah list raw CSV rows menjadi DataFrame output yang sudah dikoreksi.
    Tiap raw row menghasilkan 1 atau 2 baris output (satu per kamera).
    """
    out_cols = [
        "Date",
        "Time",
        "GPS Latitude",
        "GPS Longitude",
        "KM HM",
        "Accuracy",
        "Defect Type",
        "Image",
        "POSITION",
    ]
    records = []
    for row in raw_rows:
        if not row or not row[0].strip():
            continue
        f = _classify_and_fix_row(row)
        date, time_val = _parse_datetime(f["time_raw"])
        lat, lon = _parse_gps(f["gps_raw"])
        base = {
            "Date": date,
            "Time": time_val,
            "GPS Latitude": lat,
            "GPS Longitude": lon,
            "KM HM": "-",
            "Accuracy": f["conf_score"],
            # "Defect Type": f["condition"],
        }
        if f["left_defect"] != "-":
            records.append(
                {
                    **base,
                    "Defect Type": f["left_defect"],
                    "Image": f["left_image"],
                    "POSITION": "Left",
                }
            )
        if f["right_defect"] != "-":
            records.append(
                {
                    **base,
                    "Defect Type": f["right_defect"],
                    "Image": f["right_image"],
                    "POSITION": "Right",
                }
            )
    return pd.DataFrame(records, columns=out_cols)


def _apply_excel_style(output_path: str, n_cols: int):
    """Terapkan format plain putih + GPS merged header ke file Excel."""
    wb = load_workbook(output_path)
    ws = wb.active

    thin = Border(
        left=Side(style="thin", color="CCCCCC"),
        right=Side(style="thin", color="CCCCCC"),
        top=Side(style="thin", color="CCCCCC"),
        bottom=Side(style="thin", color="CCCCCC"),
    )
    header_border = Border(
        left=Side(style="thin", color="AAAAAA"),
        right=Side(style="thin", color="AAAAAA"),
        top=Side(style="thin", color="AAAAAA"),
        bottom=Side(style="medium", color="888888"),
    )
    header_fill = PatternFill("solid", start_color="F2F2F2", end_color="F2F2F2")
    white_fill = PatternFill("solid", start_color="FFFFFF", end_color="FFFFFF")
    alt_fill = PatternFill("solid", start_color="F9F9F9", end_color="F9F9F9")

    # Row 1 — GPS merged header
    ws.insert_rows(1)
    ws.merge_cells("C1:D1")
    ws["C1"].value = "GPS"
    ws["C1"].font = Font(name="Arial", bold=True, size=10)
    ws["C1"].alignment = Alignment(horizontal="center", vertical="center")
    ws["C1"].border = header_border
    ws["C1"].fill = header_fill
    for col in range(1, n_cols + 1):
        if col not in (3, 4):
            ws.cell(row=1, column=col).border = header_border
            ws.cell(row=1, column=col).fill = header_fill
    ws.row_dimensions[1].height = 16

    # Row 2 — sub-header
    sub_headers = [
        "Date",
        "Time",
        "Latitude",
        "Longitude",
        "KM HM",
        "Defect Type",
        "Confidence",
        "Condition",
        "Image",
        "Camera",
        "Left Total",
        "Right Total",
    ]
    for col_idx, name in enumerate(sub_headers[:n_cols], start=1):
        cell = ws.cell(row=2, column=col_idx)
        cell.value = name
        cell.font = Font(name="Arial", bold=True, size=10)
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )
        cell.border = header_border
        cell.fill = header_fill
    ws.row_dimensions[2].height = 22

    # Data rows — alternating white / near-white
    for row_idx in range(3, ws.max_row + 1):
        fill = white_fill if row_idx % 2 == 0 else alt_fill
        for col_idx in range(1, n_cols + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="center", horizontal="center")
            cell.border = thin
            cell.fill = fill

    col_widths = [13, 10, 13, 13, 9, 16, 11, 24, 32, 8, 11, 11]
    for i, w in enumerate(col_widths[:n_cols], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.freeze_panes = "A3"
    wb.save(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CSV COLUMNS (struktur yang benar untuk disimpan dari pipeline)
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "Time",  # timestamp: '20260408_200305'
    "GPS",  # raw GPS string
    "Accuracy",  # ← data aktual: Left Defect Type
    "Loss",  # ← data aktual: Left Total count
    "Confidence",  # ← data aktual: Right Total count
    "Left Defect Type",  # ← data aktual: Condition string
    "Left Total",  # ← data aktual: Confidence score
    "Left Image Name",  # posisi gambar/defect kanan bervariasi (pola A–E)
    "Right Defect Type",
    "Right Total",
    "Right Image Name",
    "Status",
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────


class ImageSaverProcess(Process):
    """
    Proses terpisah untuk menyimpan gambar dan log data ke CSV/Excel.
    Komunikasi via multiprocessing.Queue (tidak ada PyQtSignal).

    Task queue format: (task_type, data)
      - ("image",    (img_rgb: np.ndarray, filename: str))
      - ("data_log", row_data: dict)
      - ("export",   None)
      - ("stop",     None)

    CSV yang ditulis menggunakan CSV_COLUMNS di atas.
    Saat export, data CSV diperbaiki kolom-kolomnya lalu disimpan ke Excel
    dengan format plain putih dan GPS merged header.
    """

    def __init__(self, task_queue: Queue, save_dir: str = "./defect"):
        super().__init__(daemon=True, name="ImageSaverProcess")
        self.task_queue = task_queue
        self.save_dir = f"{save_dir}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        self.temp_csv = os.path.join(self.save_dir, "temp_session_data.csv")
        self.final_excel = os.path.join(self.save_dir, "laporan_kerusakan.xlsx")
        self.csv_buffer: list[dict] = []
        self.BUFFER_LIMIT = 5

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self._init_csv()
        log.info(f"ImageSaverProcess dimulai. Direktori: {self.save_dir}")

        while True:
            try:
                task_type, data = self.task_queue.get(timeout=1.0)
                if task_type == "image":
                    self._save_image(data)
                elif task_type == "data_log":
                    self._process_log_data(data)
                elif task_type == "export":
                    self._export_csv_to_excel()
                elif task_type == "stop":
                    self._export_csv_to_excel()
                    log.info("ImageSaverProcess berhenti.")
                    break
            except Exception:
                continue  # queue timeout — lanjut

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _init_csv(self):
        """Buat file CSV kosong dengan header jika belum ada."""
        if not os.path.exists(self.temp_csv):
            pd.DataFrame(columns=CSV_COLUMNS).to_csv(self.temp_csv, index=False)

    def _process_log_data(self, row_data: dict):
        self.csv_buffer.append(row_data)
        if len(self.csv_buffer) >= self.BUFFER_LIMIT:
            self._flush_buffer_to_csv()

    def _flush_buffer_to_csv(self):
        if not self.csv_buffer:
            return
        try:
            pd.DataFrame(self.csv_buffer).to_csv(
                self.temp_csv, mode="a", index=False, header=False
            )
            self.csv_buffer = []
        except Exception as e:
            log.error(f"Gagal tulis CSV: {e}")

    # ── image ─────────────────────────────────────────────────────────────────

    def _save_image(self, data):
        img_rgb, filename = data
        try:
            cv2.imwrite(os.path.join(self.save_dir, filename), img_rgb)
        except Exception as e:
            log.error(f"Gagal save image: {e}")

    # ── Excel export ──────────────────────────────────────────────────────────

    def _export_csv_to_excel(self):
        """
        Versi O(1) Formatting: Single-pass I/O, Delegated Rendering.
        """
        self._flush_buffer_to_csv()

        if not os.path.exists(self.temp_csv):
            log.warning("Tidak ada data CSV untuk diexport.")
            return

        try:
            raw_rows = self._parse_raw_csv(self.temp_csv)
            if not raw_rows:
                log.warning("CSV kosong, tidak ada data untuk diexport.")
                return

            df = _build_output_df(raw_rows)
            n_cols = len(df.columns)

            log.info(
                f"Export: {len(raw_rows)} baris CSV → "
                f"{len(df)} baris output "
                f"(Left: {(df['POSITION'] == 'Left').sum()}, "
                f"Right: {(df['POSITION'] == 'Right').sum()})"
            )

            # PENDEKATAN STRATEGIS: Tulis data dan style SEKALIGUS di memori C-level
            with pd.ExcelWriter(self.final_excel, engine="xlsxwriter") as writer:
                # 1. Tulis dataset mentah (mulai di baris ke-3 Excel, lewati baris header default pandas)
                df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Rail Inspection",
                    startrow=2,
                    header=False,
                )

                workbook = writer.book
                worksheet = writer.sheets["Rail Inspection"]

                # 2. Definisi *Style Blueprint* (Hanya dibuat 1 kali di memori, bukan per-sel)
                header_format = workbook.add_format(
                    {
                        "bold": True,
                        "align": "center",
                        "valign": "vcenter",
                        "bg_color": "#F2F2F2",
                        "border": 1,
                        "border_color": "#AAAAAA",
                        "font_name": "Arial",
                        "font_size": 10,
                    }
                )

                cell_format = workbook.add_format(
                    {
                        "align": "center",
                        "valign": "vcenter",
                        "border": 1,
                        "border_color": "#CCCCCC",
                        "font_name": "Arial",
                        "font_size": 10,
                    }
                )

                alt_fill_format = workbook.add_format({"bg_color": "#F9F9F9"})

                # 3. Terapkan style ke seluruh rentang KOLOM sekaligus (O(1) eksekusi)
                col_widths = [13, 10, 13, 13, 9, 16, 11, 24, 32, 8, 11, 11]
                for col_idx, width in enumerate(col_widths[:n_cols]):
                    worksheet.set_column(col_idx, col_idx, width, cell_format)

                # Atur tinggi baris (opsional, disesuaikan dengan estetika sebelumnya)
                worksheet.set_row(0, 16)
                worksheet.set_row(1, 22)

                # 4. Injeksi Custom Headers (Operasi manual yang cepat)
                worksheet.merge_range(0, 2, 0, 3, "GPS", header_format)  # C1:D1
                for col_idx in range(n_cols):
                    if col_idx not in (2, 3):
                        worksheet.write(0, col_idx, "", header_format)  # Sisa Baris 1
                    worksheet.write(
                        1, col_idx, df.columns[col_idx], header_format
                    )  # Baris 2

                # 5. Delegasi Render Zebra Striping ke Microsoft Excel
                # Memerintahkan software Excel untuk mengevaluasi warna tiap buka file, bukan di Python
                data_start_row = 2
                data_end_row = len(df) + 1
                worksheet.conditional_format(
                    data_start_row,
                    0,
                    data_end_row,
                    n_cols - 1,
                    {
                        "type": "formula",
                        "criteria": "=MOD(ROW(),2)=0",
                        "format": alt_fill_format,
                    },
                )

                # 6. Freeze Panes
                worksheet.freeze_panes(2, 0)

            log.info(f"Data diexport SECARA EFISIEN ke: {self.final_excel}")

        except Exception as e:
            log.error(f"Gagal Export Excel: {e}")

    @staticmethod
    def _parse_raw_csv(filepath: str) -> list:
        """
        Baca CSV baris per baris via csv.reader agar quoted commas
        (contoh: "Corrugations, Squat") tidak memecah kolom.
        Return list of rows tanpa header.
        """
        rows = []
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) == 1:
                    # baris terbungkus kutip ganda ekstra — parse ulang
                    inner = row[0].strip().strip('"')
                    row = list(csv.reader([inner]))[0]
                rows.append(row)
        return rows

    # ── public helper ─────────────────────────────────────────────────────────

    def add_task(self, task_type: str, data=None):
        """Helper untuk mengirim task dari proses lain."""
        self.task_queue.put((task_type, data))


# ============================================================================
# CLASS 4 — CameraReaderProcess
# Menggantikan CameraReader(QThread) → multiprocessing.Process
# Tidak ada pyqtSignal; frame dikirim via Queue ke BatchWorker
# ============================================================================
class CameraReaderProcess(Process):
    """
    Proses pembaca kamera. Frame mentah dikirim ke `frame_queue`.

    frame_queue format: (camera_id: int, frame: bytes)
      Frame di-serialize ke bytes karena numpy array tidak bisa langsung
      dikirim lintas proses (perlu pickle-friendly format).
    """

    def __init__(
        self,
        camera_id: int,
        source,  # int (device) atau str (path video)
        frame_queue: Queue,
        control_queue: Queue,
        broadcast_queue: Queue,
        fps: int = 30,
    ):
        super().__init__(daemon=True, name=f"CameraReaderProcess-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.frame_queue = frame_queue
        self.control_queue = control_queue
        self.broadcast_queue = broadcast_queue
        self.fps = fps

    def run(self):
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        from ultralytics import YOLO

        cap = cv2.VideoCapture(self.source)
        # if not cap.isOpened():
        #     log.error(f"Gagal membuka kamera {self.camera_id}: {self.source}")
        #     return

        log.info(f"CameraReaderProcess-{self.camera_id} dimulai.")
        frame_duration = 1.0 / self.fps
        paused = False
        fail_count = 0
        is_disconnected = False

        while True:
            # Cek perintah kontrol (non-blocking)
            try:
                cmd = self.control_queue.get_nowait()
                action = cmd.get("action", "")
                if action == "stop":
                    break
                elif action == "pause":
                    paused = True
                elif action == "resume":
                    paused = False
                elif action == "seek":
                    seconds = cmd.get("seconds", 0)
                    current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    target_ms = max(0, current_ms + seconds * 1000)
                    cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
            except Exception:
                pass

            if paused:
                time.sleep(0.1)
                continue

            t0 = time.time()
            ret = False
            if cap.isOpened():
                ret, frame = cap.read()

            if not ret:
                fail_count += 1
                # Jika gagal baca selama ~1 detik (30 frames)
                if fail_count == 30 and not is_disconnected:
                    is_disconnected = True
                    self.broadcast_queue.put_nowait(
                        {
                            "type": "camera_error",
                            "camera_id": self.camera_id,
                            "status": "disconnected",
                        }
                    )
                    # Re-initialize camera
                    cap.release()

                # Auto retry setiap detik
                if is_disconnected and fail_count % 30 == 0:
                    cap = cv2.VideoCapture(self.source)

                time.sleep(0.01)
                continue

            # Jika berhasil setelah putus
            if is_disconnected:
                is_disconnected = False
                fail_count = 0
                self.broadcast_queue.put_nowait(
                    {
                        "type": "camera_error",
                        "camera_id": self.camera_id,
                        "status": "connected",
                    }
                )

            fail_count = 0

            # Kirim frame via Queue (numpy array → pickle otomatis oleh mp.Queue)
            try:
                # Drop frame lama jika queue penuh (hindari lag akumulasi)
                if self.frame_queue.qsize() < 4:
                    self.frame_queue.put_nowait((self.camera_id, frame))
            except Exception:
                pass

            elapsed = time.time() - t0
            wait = frame_duration - elapsed
            if wait > 0:
                time.sleep(wait)

        cap.release()
        log.info(f"CameraReaderProcess-{self.camera_id} berhenti.")


# ============================================================================
# CLASS 4.5 — GPSReaderProcess (BARU)
# ============================================================================


class GPSReaderProcess(Process):
    """
    Proses mandiri untuk membaca modul GPS via Serial Port.
    Berjalan independen dari model AI sehingga status GPS selalu terkirim.
    """

    def __init__(
        self,
        broadcast_queue: Queue,
        internal_gps_queue: Queue,
        port="/dev/ttyTHS1",
        baudrate=9600,
    ):
        super().__init__(daemon=True, name="GPSReaderProcess")
        self.broadcast_queue = broadcast_queue
        self.internal_gps_queue = internal_gps_queue
        self.port = port
        self.baudrate = baudrate

    def run(self):
        import serial
        import pynmea2
        import time
        import queue

        log.info(f"GPSReaderProcess dimulai. Menunggu sinyal di {self.port}...")

        while True:
            try:
                # Buka koneksi serial
                with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                    while True:
                        line = ser.readline().decode("ascii", errors="replace").strip()

                        # Filter untuk GPGGA (GPS biasa) atau GNGGA (GNSS multi-satelit)
                        if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                            try:
                                msg = pynmea2.parse(line)

                                # Cek apakah satelit sudah terkunci (Kualitas > 0)
                                if msg.gps_qual > 0:
                                    lat = f"{msg.latitude:.6f}"
                                    lon = f"{msg.longitude:.6f}"
                                    gps_text = f"Latitude: {lat}\nLongitude: {lon}"

                                    # 1. Kirim ke Frontend untuk update UI Real-time
                                    try:
                                        self.broadcast_queue.put_nowait(
                                            {
                                                "type": "gps_status",
                                                "status": "fixed",
                                                "lat": lat,
                                                "lon": lon,
                                            }
                                        )
                                    except queue.Full:
                                        pass

                                    # 2. Kirim ke internal queue untuk dicatat BatchWorker ke CSV
                                    # Kosongkan queue lama agar selalu mendapat data paling fresh
                                    while not self.internal_gps_queue.empty():
                                        try:
                                            self.internal_gps_queue.get_nowait()
                                        except queue.Empty:
                                            break
                                    self.internal_gps_queue.put_nowait(gps_text)

                                else:
                                    # Belum terkunci (masih mencari satelit)
                                    try:
                                        self.broadcast_queue.put_nowait(
                                            {
                                                "type": "gps_status",
                                                "status": "searching",
                                                "message": "GPS: Mencari Satelit (Pastikan Open Sky)...",
                                            }
                                        )
                                    except queue.Full:
                                        pass

                            except pynmea2.ParseError:
                                pass  # Abaikan jika ada baris NMEA yang terpotong/rusak

            except serial.SerialException as e:
                # Jika port tidak ditemukan atau akses ditolak
                log.error(
                    f"Error port serial GPS ({self.port}): {e}. Retrying dalam 3 detik..."
                )
                try:
                    self.broadcast_queue.put_nowait(
                        {
                            "type": "gps_status",
                            "status": "searching",
                            "message": f"Error GPS: Port {self.port} tidak dapat diakses.",
                        }
                    )
                except queue.Full:
                    pass
                time.sleep(3)  # Tunggu sebelum mencoba koneksi ulang

            except Exception as e:
                log.error(f"Error tak terduga pada GPSReaderProcess: {e}")
                time.sleep(3)


# ============================================================================
# CLASS 5 — BatchWorkerProcess
# Menggantikan BatchWorker(QThread) → multiprocessing.Process
# Input: frame_queue | Output: broadcast_queue + saver_queue
# ============================================================================
class BatchWorkerProcess(Process):
    def __init__(
        self,
        frame_queue: Queue,
        broadcast_queue: Queue,
        saver_queue: Queue,
        control_queue: Queue,
        internal_gps_queue: Queue,  # TAMBAHKAN INI
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        threshold_queue=200,
    ):
        super().__init__(daemon=True, name="BatchWorkerProcess")
        self.frame_queue = frame_queue
        self.broadcast_queue = broadcast_queue
        self.saver_queue = saver_queue
        self.control_queue = control_queue
        self.internal_gps_queue = internal_gps_queue
        self.model_name = model_name
        self.threshold_queue = threshold_queue
        self.saver_queue = saver_queue

    def run(self):
        # Import di sini agar tidak perlu di-serialize lintas proses
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        from ultralytics import YOLO

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"BatchWorkerProcess dimulai. Device: {device}")

        processor = SegformerImageProcessor.from_pretrained(self.model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        model.to(device).eval()

        yolo_model = YOLO("./final_model_v2/penambat.pt").to(device)

        frame_buffer: dict[int, np.ndarray] = {}
        graph_started = False
        paused = False

        # Mode awal "standby" — tidak ada model berjalan sampai user buka page
        # Valid values: "segformer" | "yolo" | "standby"
        active_model = "standby"

        current_gps_string = "GPS: Menunggu satelit..."
        # "GPS": "Latitude: -6.975971\nLongitude: 107.629658",
        # current_gps_string = "Latitude: N/A\nLongitude:N/A"

        current_threshold = self.threshold_queue

        while True:
            # Cek perintah kontrol
            try:
                cmd = self.control_queue.get_nowait()
                action = cmd.get("action", "")
                if action == "export":
                    self.saver_queue.put(("export", None))
                    log.info(
                        "UI meminta Ekspor: Perintah diteruskan ke ImageSaverProcess"
                    )
                elif action == "set_threshold":
                    current_threshold = int(cmd.get("value", 1))
                elif action == "stop":
                    break
                elif action == "pause":
                    paused = True
                elif action == "resume":
                    paused = False
                elif action == "set_mode":
                    new_mode = cmd.get("mode", "standby")
                    if new_mode != active_model:
                        log.info(f"Mode model berubah: {active_model} → {new_mode}")
                        active_model = new_mode

            except Exception:
                pass

            if paused:
                time.sleep(0.05)
                continue

            try:
                current_gps_string = self.internal_gps_queue.get_nowait()
            except Exception:
                pass

            # Jika mode standby, kuras frame_queue agar tidak menumpuk lalu idle
            if active_model == "standby":
                try:
                    self.frame_queue.get_nowait()
                except Exception:
                    pass
                time.sleep(0.02)
                continue

            # Ambil frame
            try:
                camera_id, frame = self.frame_queue.get(timeout=0.5)
            except Exception:
                continue

            frame_buffer[camera_id] = frame

            # Proses batch saat kedua kamera tersedia
            if 1 in frame_buffer and 2 in frame_buffer:
                frame1 = frame_buffer.pop(1)
                frame2 = frame_buffer.pop(2)

                try:
                    results, log_data, images = self._process_batch(
                        frame1,
                        frame2,
                        processor,
                        model,
                        yolo_model,
                        device,
                        active_model,
                        current_gps_string,  # PASSING DATA GPS KE FUNGSI INI
                        current_threshold,
                    )

                    # Kirim frame ke broadcaster (WebSocket)
                    for msg in results:
                        try:
                            self.broadcast_queue.put_nowait(msg)
                        except Exception:
                            pass

                    # Kirim log + gambar ke saver
                    if log_data["Left Total"] > 0 or log_data["Right Total"] > 0:
                        for img_data in images:
                            self.saver_queue.put(("image", img_data))
                        self.saver_queue.put(("data_log", log_data))

                    # Kirim sinyal graph sekali
                    if not graph_started:
                        self.broadcast_queue.put_nowait({"type": "graph_start"})
                        graph_started = True

                except Exception as e:
                    log.error(f"Error proses batch: {e}")

        log.info("BatchWorkerProcess berhenti.")

    def _process_batch(
        self,
        frame_bgr1,
        frame_bgr2,
        processor,
        model,
        yolo_model,
        device,
        active_model,
        current_gps_string,
        current_threshold,
    ):
        roi_inputs = []
        meta_data = []
        threshold = 200 if current_threshold is None else current_threshold

        # (Logika crop ROI tetap berjalan karena dibutuhkan untuk base frame)
        for frame in [frame_bgr1, frame_bgr2]:
            """ ROTATE GAMBAR DEFECT """
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_flipped = cv2.flip(frame, 0)
            final_frame = frame_flipped.copy()
            h, w = frame_flipped.shape[:2]
            w_third = w // 3
            w_start, w_end = w_third, w_third * 2
            frame_cropped_w = frame_flipped[:, w_start:w_end]
            h_half = frame_cropped_w.shape[0] // 2
            frame_roi = frame_cropped_w[h_half:, :]

            meta_data.append(
                {
                    "final_frame": final_frame,
                    "w_range": (w_start, w_end),
                    "h_start": h_half,
                    "roi_shape": frame_roi.shape[:2],
                }
            )
            roi_inputs.append(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB))

        # ==========================================
        # KONDISI 1: JALANKAN SEGFORMER
        # ==========================================
        seg_results = None
        if active_model == "segformer":
            inputs = processor(images=roi_inputs, return_tensors="pt").to(device)
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                seg_results = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[m["roi_shape"] for m in meta_data]
                )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_data = {
            # ... Data GPS/Log inisialisasi default tetap sama ...
            "Time": timestamp,
            # "GPS": "Latitude: -6.975971\nLongitude: 107.629658",
            "GPS": current_gps_string,
            "Left Defect Type": "-",
            "Left Total": 0,
            "Right Total": 0,
            "Status": "Aman",
            "Accuracy": "0.9500",
        }

        broadcast_msgs = []
        images_to_save = []

        for i in range(2):
            camera_id = i + 1
            meta = meta_data[i]
            side = "Left" if camera_id == 1 else "Right"
            final_frame = meta["final_frame"]
            w_start, w_end = meta["w_range"]
            h_start = meta["h_start"]

            total_contours_all = 0
            detected_classes = []
            clip_counts = {"DE-Clip": 0, "E-Clip": 0, "KA-Clip": 0, "No Clip": 0}

            # ==========================================
            # KONDISI 2: JALANKAN YOLO
            # ==========================================
            if active_model == "yolo":
                yolo_results = yolo_model(final_frame, verbose=False)
                for r in yolo_results:
                    for box in r.boxes:
                        cls_name = yolo_model.names[int(box.cls[0])]
                        if cls_name in clip_counts:
                            clip_counts[cls_name] += 1
                    final_frame = r.plot()

            # ==========================================
            # EKSTRAKSI SEGFORMER (JIKA ADA)
            # ==========================================
            if seg_results is not None:
                mask_np = seg_results[i].cpu().numpy().astype(np.uint8)
                result_view_roi = final_frame[h_start:, w_start:w_end].copy()

                for class_id in np.unique(mask_np):
                    if class_id in SKIP_CLASSES:
                        continue
                    class_id_int = int(class_id)
                    class_label = CLASS_NAMES.get(class_id_int, f"Class {class_id_int}")
                    color = CLASS_COLORS.get(class_id_int, (255, 255, 255))

                    mask_class = (mask_np == class_id).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    selected_boxes = [
                        cv2.boundingRect(c)
                        for c in contours
                        # if cv2.boundingRect(c)[3] > 230
                        if cv2.boundingRect(c)[3] > threshold
                    ]

                    if not selected_boxes:
                        continue

                    for x, y, cw, ch in selected_boxes:
                        cv2.rectangle(
                            result_view_roi, (x, y), (x + cw, y + ch), color, 2
                        )

                    class_count = len(selected_boxes)
                    total_contours_all += class_count
                    detected_classes.append({"name": class_label, "count": class_count})

                final_frame[h_start:, w_start:w_end] = result_view_roi

            # Siapkan Frame Output
            final_frame_resized = cv2.resize(final_frame, (360, 640))
            # final_output_rgb = cv2.cvtColor(
            #     cv2.flip(final_frame_resized, 0), cv2.COLOR_RGB2BGR
            # )
            final_output_bgr = cv2.flip(final_frame_resized, 0)

            _, buf = cv2.imencode(
                ".jpg", final_output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            frame_b64 = base64.b64encode(buf).decode("utf-8")

            broadcast_msgs.append(
                {
                    "type": "frame" if active_model == "segformer" else "frame_yolo",
                    "camera_id": camera_id,
                    "frame_b64": frame_b64,
                    "total_contours": total_contours_all,
                    "gps": log_data["GPS"],
                    "detected_classes": detected_classes,
                    "timestamp": timestamp,
                    "inspection": clip_counts,  # Data YOLO dikirim ke UI
                }
            )

            if total_contours_all > 0:
                log_data[f"{side} Defect Type"] = ", ".join(
                    c["name"] for c in detected_classes
                )
                log_data[f"{side} Total"] = total_contours_all
                filename = f"{timestamp}_cam{camera_id}.jpg"
                log_data[f"{side} Image Name"] = filename
                images_to_save.append((final_output_bgr, filename))

        # Status
        has_left = log_data["Left Total"] > 0
        has_right = log_data["Right Total"] > 0
        if has_left and has_right:
            log_data["Status"] = "Kiri & Kanan Perlu Perbaikan"
        elif has_left:
            log_data["Status"] = "Kiri Perlu Perbaikan"
        elif has_right:
            log_data["Status"] = "Kanan Perlu Perbaikan"

        # Kirim log ke broadcaster juga (untuk tampilan tabel di web)
        broadcast_msgs.append(
            {
                "type": "log",
                "data": {k: v for k, v in log_data.items() if k != "GPS"},
                "timestamp": timestamp,
            }
        )

        return broadcast_msgs, log_data, images_to_save


# ============================================================================
# FANOUT — harus di level module agar bisa di-pickle oleh spawn
# ============================================================================


def _fanout_worker(
    master_ctrl: Queue, ctrl_batch: Queue, ctrl_cam1: Queue, ctrl_cam2: Queue
):
    """Meneruskan perintah dari master_ctrl ke semua proses kontrol."""
    while True:
        try:
            cmd = master_ctrl.get(timeout=1.0)
            for q in [ctrl_batch, ctrl_cam1, ctrl_cam2]:
                try:
                    q.put_nowait(cmd)
                except Exception:
                    pass
        except Exception:
            pass


def main(
    cam1_source=0,
    cam2_source=1,
    model_name: str = "./final_model_v2/",
    web_host: str = "0.0.0.0",
    web_port: int = 8000,
):
    """
    Jalankan seluruh pipeline:
      1. ImageSaverProcess   — menyimpan gambar & CSV/Excel
      2. CameraReaderProcess × 2 — baca kamera kiri & kanan
      3. BatchWorkerProcess  — AI inference
      4. WebServer           — FastAPI + WebSocket (berjalan di main process)
    """
    # --- Shared Queues ---
    saver_queue = Queue(maxsize=50)  # → ImageSaverProcess
    frame_queue = Queue(maxsize=8)  # CameraReader → BatchWorker
    broadcast_queue = Queue(maxsize=100)  # BatchWorker  → WebBroadcaster
    ctrl_batch = Queue(maxsize=10)  # Web → BatchWorker
    ctrl_cam1 = Queue(maxsize=10)  # Web → CamReader-1
    ctrl_cam2 = Queue(maxsize=10)  # Web → CamReader-2
    internal_gps_queue = Queue(maxsize=2)

    # Semua control queue dalam satu proxy queue yang di-fanout
    master_ctrl = Queue(maxsize=20)

    # --- Proses ---
    saver = ImageSaverProcess(saver_queue, "./data_defect/defect")
    gps_reader = GPSReaderProcess(broadcast_queue, internal_gps_queue)

    cam1 = CameraReaderProcess(1, cam1_source, frame_queue, ctrl_cam1, broadcast_queue)
    cam2 = CameraReaderProcess(2, cam2_source, frame_queue, ctrl_cam2, broadcast_queue)

    worker = BatchWorkerProcess(
        frame_queue,
        broadcast_queue,
        saver_queue,
        ctrl_batch,
        internal_gps_queue,
        model_name,
    )

    # --- Web Server ---
    broadcaster = WebBroadcaster(broadcast_queue)
    server = WebServer(broadcaster, master_ctrl, host=web_host, port=web_port)

    # --- Fan-out perintah dari web ke semua proses ---
    fanout_proc = Process(
        target=_fanout_worker,
        args=(master_ctrl, ctrl_batch, ctrl_cam1, ctrl_cam2),
        daemon=True,
        name="FanoutProcess",
    )

    # --- Start ---
    log.info("Memulai semua proses...")
    saver.start()
    gps_reader.start()
    cam1.start()
    cam2.start()
    worker.start()
    fanout_proc.start()

    log.info(f"Dashboard tersedia di: http://{web_host}:{web_port}")
    log.info("Tekan Ctrl+C untuk berhenti.")

    try:
        # WebServer blocking — berjalan di main process (asyncio event loop)
        server.run()
    except KeyboardInterrupt:
        log.info("Menghentikan semua proses...")
    finally:
        log.info("Mengirim sinyal stop ke semua proses...")
        master_ctrl.put({"action": "stop"})

        # Beri tahu saver untuk berhenti (ini akan memicu _export_csv_to_excel)
        saver_queue.put(("stop", None))

        # Tunggu proses saver menyelesaikan ekspor (beri waktu yang masuk akal, misal 15 detik)
        log.info("Menunggu ImageSaverProcess menyelesaikan ekspor Excel...")
        saver.join(timeout=30)

        if saver.is_alive():
            log.warning("ImageSaverProcess memakan waktu terlalu lama. Membunuh paksa.")
            saver.terminate()
        else:
            log.info("Ekspor Excel selesai dengan aman.")

        # Matikan proses sisanya
        for p in [worker, cam1, cam2, fanout_proc]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)


if __name__ == "__main__":
    # Guard wajib untuk multiprocessing di Windows/macOS
    mp.set_start_method("spawn", force=True)
    main(
        # cam1_source="/dev/cam_kiri",
        # cam2_source="/dev/cam_kanan",
        cam1_source="../../combine-app/kiri2.mp4",
        cam2_source="../../combine-app/kanan2.mp4",
        web_host="0.0.0.0",
        web_port=8000,
    )
