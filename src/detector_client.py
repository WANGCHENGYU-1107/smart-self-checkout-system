# -*- coding: utf-8 -*-
"""
YOLO 影像辨識 → 同步到你的 POS（/api/cart/add）
------------------------------------------------
相容你的 pos.py/server.py 設計：
- 讀 products.csv（name,price）
- 讀 size_variants.csv（base_name,small_name,large_name,split_g）
- 如果是「需要秤重」的 base_name → 呼叫 server 的 /api/scale/request
- 否則直接 /api/cart/add 送單
- 視窗標籤使用支援中文的 draw_label_smart，不會被框吃掉

快捷鍵：
- q：離開
- s：切換「自動送單」(auto_push) 開/關（預設開）
- f：全螢幕/還原（畫面 letterbox）
- c：強制解除鎖定（方便快速換商品）
"""

from ultralytics import YOLO
import cv2, os, time, socket, csv, json
from collections import deque, Counter
import numpy as np
from pathlib import Path
import requests
from typing import Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

# ====== 基本參數 ======
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "best.pt")
IMGSZ      = 640
CONF_TH    = 0.70
IOU_TH     = 0.50
WIN_NAME   = "YOLO Detector"      # 純英文，避免亂碼
CAM_INDEX  = 1                    # 0=內建、1=外接（打不開會自動嘗試另一個）

# —— 穩定器（多幀投票） ——
WINDOW     = 10                   # 最近 N 幀做投票
VOTE_N     = 5                    # 至少 N 票視為穩定
ACCEPT_TH  = 0.85                 # 多數票比例門檻
REVOKE_TH  = 0.80                 # 比例降到此值以下解除鎖定
CLEAR_FRAMES               = 20   # 目標消失 N 幀才解除鎖定
JUST_UNLATCHED_SHOW_FRAMES = 45   # 解除鎖定提示顯示幾幀

# —— 誤判過濾（中心優先＋面積＋每幀信心＋類別下限） ——
CENTER_FOCUS       = 0.80   # 只收畫面中央 60% 區域
MIN_BOX_AREA_RATIO = 0.02   # 框面積至少佔全畫面的 2%
PER_FRAME_CONF_TH  = 0.85   # 單幀最低信心
CLASS_MIN_CONF = {          # 某些易混淆類別拉高門檻（用映射後的中文名）
    "奇多"  : 0.90,
}

# ========== 中文字型與聰明標籤 ==========
def _pick_cn_font() -> str | None:
    candidates = [
        str(BASE_DIR / "NotoSansCJK-Regular.ttc"),
        "C:/Windows/Fonts/msjh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

CN_FONT_PATH = _pick_cn_font()

def draw_label_smart(img: np.ndarray, text: str, x1: int, y1: int, x2: int, y2: int,
                     color_bgr: tuple[int,int,int] = (0,0,0)) -> None:
    """標籤自動找不出界位置（上→下→框內），並依框大小調字體大小。"""
    H, W = img.shape[:2]
    pad = 4
    box_h = max(1, y2 - y1)
    font_px = int(np.clip(box_h // 8, 16, 28))

    if CN_FONT_PATH:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = ImageFont.truetype(CN_FONT_PATH, size=font_px)
        bx = draw.textbbox((0, 0), text, font=font)
        tw, th = (bx[2] - bx[0], bx[3] - bx[1])

        cand = [(x1, y1 - th - pad*2), (x1, y2 + pad), (x1, y1 + pad)]
        tx, ty = 0, 0
        for cx, cy in cand:
            if 0 <= cx and cx + tw + pad*2 <= W and 0 <= cy and cy + th + pad*2 <= H:
                tx, ty = cx, cy
                break
        else:
            tx = int(np.clip(x1, 0, max(0, W - tw - pad*2)))
            ty = int(np.clip(y1, 0, max(0, H - th - pad*2)))

        draw.rectangle([tx, ty, tx + tw + pad*2, ty + th + pad*2], fill=(255,255,255,220))
        r,g,b = color_bgr[2], color_bgr[1], color_bgr[0]
        draw.text((tx + pad, ty + pad), text, font=font, fill=(r,g,b))
        img[:, :, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return

    # 無中文字型就退回 OpenCV 英文字體
    font_scale = max(0.5, min(1.0, box_h / 200.0))
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cand = [(x1, y1 - th - pad), (x1, y2 + th + pad), (x1, y1 + th + pad)]
    tx, ty = 0, 0
    for cx, cy in cand:
        top = cy - th - pad
        bot = cy + pad
        if 0 <= cx and cx + tw + pad*2 <= W and 0 <= top and bot <= H:
            tx, ty = cx, cy
            break
    else:
        tx = int(np.clip(x1, 0, max(0, W - tw - pad*2)))
        ty = int(np.clip(y1 + th + pad, 0, H))
    cv2.rectangle(img, (tx, ty - th - pad), (tx + tw + pad*2, ty + pad), (255,255,255), -1)
    cv2.putText(img, text, (tx + pad, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness, cv2.LINE_AA)

# 舊程式相容：固定座標小白底標籤（用在「解除鎖定」提示）
def draw_label(img, text, x, y, color_bgr=(0,0,0)):
    H, W = img.shape[:2]
    pad = 4
    if CN_FONT_PATH:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = ImageFont.truetype(CN_FONT_PATH, size=20)
        bx = draw.textbbox((0, 0), text, font=font)
        tw, th = bx[2] - bx[0], bx[3] - bx[1]
        tx = int(np.clip(x, 0, max(0, W - tw - pad*2)))
        ty = int(np.clip(y, 0, max(0, H - th - pad*2)))
        draw.rectangle([tx, ty, tx + tw + pad*2, ty + th + pad*2], fill=(255,255,255,220))
        r,g,b = color_bgr[2], color_bgr[1], color_bgr[0]
        draw.text((tx + pad, ty + pad), text, font=font, fill=(r,g,b))
        img[:, :, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        font_scale = 0.6
        thickness  = 2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = int(np.clip(x, 0, max(0, W - tw - pad*2)))
        ty = int(np.clip(y + th + pad, 0, H))
        cv2.rectangle(img, (tx, ty - th - pad), (tx + tw + pad*2, ty + pad), (255,255,255), -1)
        cv2.putText(img, text, (tx + pad, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness, cv2.LINE_AA)

# ====== 視窗 letterbox ======
def letterbox(img: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale = min(dst_w / w, dst_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    x = (dst_w - new_w) // 2
    y = (dst_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas

# ====== 伺服器位址 & 檔案路徑 ======
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

SERVER = os.environ.get("POS_SERVER", f"http://{get_local_ip()}:5000")

def pick_csv(name: str) -> Path:
    for p in [BASE_DIR / name, BASE_DIR / "data" / name]:
        if p.exists():
            return p
    return BASE_DIR / name

PRICE_CSV     = pick_csv("products.csv")
SIZE_RULE_CSV = pick_csv("size_variants.csv")

def load_price_map(csv_path: Path) -> Dict[str, float]:
    price_map: Dict[str, float] = {}
    if not csv_path.exists(): return price_map
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            name = (row.get("name") or "").strip()
            try:
                price_map[name.lower()] = float(row.get("price"))
            except Exception:
                pass
    return price_map

def load_size_rules(csv_path: Path) -> Dict[str, Dict]:
    rules: Dict[str, Dict] = {}
    if not csv_path.exists(): return rules
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            base  = (row.get("base_name")  or "").strip()
            small = (row.get("small_name") or "").strip()
            large = (row.get("large_name") or "").strip()
            try:
                split = float(row.get("split_g"))
            except Exception:
                split = None
            if base and small and large and split:
                rules[base] = {"small": small, "large": large, "split_g": split}
    return rules

PRICE_MAP  = load_price_map(PRICE_CSV)
SIZE_RULES = load_size_rules(SIZE_RULE_CSV)

# ====== 名稱對應（raw → POS 名稱）======
NAME_MAP_RAW: Dict[str, str] = {
    # 依你的模型輸出調整
    "karamucho": "咔啦姆久",
    "cadina":    "卡迪那",
    "doritos":   "多力多滋",
    "orange green tea": "純喫茶 香橙綠茶",
    "Wheat Milk Tea":   "麥香奶茶",
    "Real Leaf":        "原萃綠茶",
    "Kuai Kuai":        "乖乖",
    "Apple Sidra":      "蘋果西打",
    "Jenyowe Squid Cracker":        "真魷味",
    "Cheetos":  "奇多",
    "Lays" :    "樂事",
}
NAME_MAP: Dict[str, str] = {k.lower().strip(): v for k, v in NAME_MAP_RAW.items()}

# ====== 同步到 POS / 啟動秤重 ======
def lookup_price(name: str) -> float:
    return PRICE_MAP.get(name.lower(), 0.0)

def push_to_pos(name: str, price: float, qty: int = 1) -> bool:
    try:
        url = f"{SERVER}/api/cart/add"
        payload = {"name": name, "price": float(price), "qty": int(qty)}
        r = requests.post(url, json=payload, timeout=1.2)
        return r.ok
    except Exception:
        return False

def request_scale(base_name: str) -> Tuple[bool, str]:
    """通知 server 啟動秤重流程。"""
    try:
        url = f"{SERVER}/api/scale/request"
        r = requests.post(url, json={"base_name": base_name}, timeout=1.5)
        if r.ok:
            return True, "ok"
        else:
            try:
                d = r.json()
                return False, d.get("error", f"http {r.status_code}")
            except Exception:
                return False, f"http {r.status_code}"
    except Exception as e:
        return False, str(e)

# ====== 穩定器 ======
class StableLatch:
    """多幀投票，鎖定送一次；消失 CLEAR_FRAMES 幀解除。"""
    def __init__(self):
        self.hist = deque(maxlen=WINDOW)
        self.latched_label = None
        self.missing_frames = 0
        self.just_unlatched_frames = 0

    def update(self, label: str):
        self.hist.append(label or "")
        c = Counter([x for x in self.hist if x])
        total = sum(c.values())
        top_label, top_cnt = (None, 0)
        if c:
            top_label, top_cnt = c.most_common(1)[0]
        ratio = (top_cnt / total) if total else 0.0

        if self.latched_label:
            if label == self.latched_label:
                self.missing_frames = 0
            else:
                self.missing_frames += 1
                if self.missing_frames >= CLEAR_FRAMES:
                    self.latched_label = None
                    self.just_unlatched_frames = JUST_UNLATCHED_SHOW_FRAMES
            return None

        if top_label and top_cnt >= VOTE_N and ratio >= ACCEPT_TH:
            self.latched_label = top_label
            self.missing_frames = 0
            return top_label
        return None

# ====== 主程式 ======
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型檔：{MODEL_PATH}")

    print(f"[Detector] 伺服器：{SERVER}")
    print(f"[Detector] 價目表：{PRICE_CSV}（共 {len(PRICE_MAP)} 筆）")
    print(f"[Detector] 秤重規則：{SIZE_RULE_CSV}（共 {len(SIZE_RULES)} 筆）")

    model = YOLO(MODEL_PATH)

    # 開相機：主 index 失敗就換另一個
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        alt_idx = 0 if CAM_INDEX != 0 else 1
        cap = cv2.VideoCapture(alt_idx, cv2.CAP_ANY)
        if cap.isOpened():
            print(f"[Detector] 主 index 開啟失敗，改用 index={alt_idx}")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    # 暖機：最多 1.5 秒
    warm_ok = False
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            warm_ok = True
            break
        time.sleep(0.05)
    if not warm_ok:
        print("[Detector] 相機開啟了，但讀不到影像（可能被占用或權限未開）。")
        cap.release()
        return

    latch = StableLatch()
    fps_t0, fps_cnt = time.time(), 0
    fps_val = 0.0
    auto_push = True
    fullscreen = False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[Detector] 讀取相機失敗")
            break

        # 推論（禁止 Ultralytics 自開視窗）
        results = model.predict(source=frame, imgsz=IMGSZ, conf=CONF_TH, iou=IOU_TH,
                                verbose=False, show=False)
        boxes = results[0].boxes

        # 當前幀候選
        current_label = ""
        raw_best = ""
        best_conf = 0.0
        best_xyxy = None

        if boxes is not None and len(boxes) > 0:
            H, W = frame.shape[:2]
            cx_min, cx_max = W * (0.5 - CENTER_FOCUS/2), W * (0.5 + CENTER_FOCUS/2)
            cy_min, cy_max = H * (0.5 - CENTER_FOCUS/2), H * (0.5 + CENTER_FOCUS/2)
            min_area = MIN_BOX_AREA_RATIO * W * H

            for b in boxes:
                cls_id = int(b.cls.item())
                conf   = float(b.conf.item())
                xyxy   = b.xyxy.cpu().numpy().astype(int).tolist()[0]
                raw    = results[0].names.get(cls_id, str(cls_id))
                mapped = NAME_MAP.get(str(raw).lower().strip(), str(raw))

                x1, y1, x2, y2 = xyxy
                area = max(0, x2 - x1) * max(0, y2 - y1)
                cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
                center_ok = (cx_min <= cx <= cx_max) and (cy_min <= cy <= cy_max)
                area_ok   = area >= min_area
                conf_ok   = conf >= PER_FRAME_CONF_TH
                class_ok  = conf >= CLASS_MIN_CONF.get(mapped, 0.0)
                if not (center_ok and area_ok and conf_ok and class_ok):
                    continue

                if conf > best_conf:
                    best_conf  = conf
                    current_label = mapped
                    raw_best      = str(raw)
                    best_xyxy     = xyxy

        # 更新穩定器
        trigger_label = latch.update(current_label)

        # 視覺化
        disp = frame.copy()
        if best_xyxy is not None:
            x1, y1, x2, y2 = best_xyxy
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_label_smart(disp, f"{raw_best} → {current_label} {best_conf:.2f}", x1, y1, x2, y2)

        # FPS 與自動送單狀態
        fps_cnt += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps_val = fps_cnt / dt
            fps_cnt = 0
            fps_t0 = time.time()
        cv2.putText(disp, f"FPS: {fps_val:.1f}  auto_push: {'ON' if auto_push else 'OFF'}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        if latch.just_unlatched_frames > 0:
            draw_label(disp, "解除鎖定", 10, 50, color_bgr=(0,0,255))
            latch.just_unlatched_frames -= 1

        # letterbox 塞滿視窗
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
            to_show = letterbox(disp, max(320, win_w), max(240, win_h))
        except Exception:
            to_show = disp
        cv2.imshow(WIN_NAME, to_show)

        # 鍵盤控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            auto_push = not auto_push
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WIN_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            )
        elif key == ord('c'):
            latch.latched_label = None
            latch.missing_frames = 0
            latch.hist.clear()
            latch.just_unlatched_frames = JUST_UNLATCHED_SHOW_FRAMES

        # —— 送單 / 啟動秤重 —— 
        if trigger_label and auto_push:
            label = trigger_label
            if label in SIZE_RULES:
                ok, err = request_scale(label)
                if ok:
                    print(f"[Detector] {label} 需要秤重 → 已通知伺服器開始。請顧客把商品放上秤。")
                else:
                    print(f"[Detector] 無法聯絡伺服器啟動秤重：{err}")
            else:
                price = lookup_price(label)
                if price <= 0:
                    print(f"[Detector] 價格未知：{label}（未送出）")
                else:
                    ok = push_to_pos(label, price, 1)
                    print(f"[Detector] 送出 → {label} x1 @ {price}：{'OK' if ok else 'FAIL'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
