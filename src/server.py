# server.py —— Flask-SocketIO（threading/long-polling），Python 3.13 相容
# -*- coding: utf-8 -*-

import os, io, re, csv, time, socket, threading
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Dict, Any, Optional

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_socketio import SocketIO
import qrcode

# ReportLab：PDF 產生與字型
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader

# ---------------- 基本設定 ----------------
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder="static")
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",   # 前端以 polling 連接
    ping_interval=10,
    ping_timeout=20,
)

cart_lock = Lock()

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

PUBLIC_BASE_URL = f"http://{get_local_ip()}:5000"

def pick_csv(name: str) -> Path:
    for p in (BASE_DIR / name, BASE_DIR / "data" / name):
        if p.exists():
            return p
    return BASE_DIR / name

PRODUCTS_CSV = pick_csv("products.csv")
VARIANTS_CSV = pick_csv("size_variants.csv")

# ---------------- 載入資料 ----------------
def load_products(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            for row in r:
                name = (row.get("name") or "").strip()
                try:
                    price = float(row.get("price"))
                except:
                    continue
                if name:
                    out[name] = price
    return out

def load_variants(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            for row in r:
                base  = (row.get("base_name")  or "").strip()
                small = (row.get("small_name") or "").strip()
                large = (row.get("large_name") or "").strip()
                try:
                    split_g = float(row.get("split_g"))
                except:
                    split_g = None
                if base and small and large and split_g is not None:
                    out[base] = {"small": small, "large": large, "split_g": float(split_g)}
    return out

PRODUCTS = load_products(PRODUCTS_CSV)
VARIANTS = load_variants(VARIANTS_CSV)
print(f"[Server] 讀到 {PRODUCTS_CSV}（{len(PRODUCTS)} 筆）")
print(f"[Server] 讀到 {VARIANTS_CSV}（{len(VARIANTS)} 筆）")
print("[Server] VARIANTS keys:", list(VARIANTS.keys()))

# ---------------- 全域狀態 ----------------
cart_state: Dict[str, Any] = {
    "items": [],
    "subtotal": 0.0,
    "discount": 0.0,
    "total": 0.0,
    "checkout_pending": False,
    "scale_pending": False,
    "scale_base": None,
    "updated_at": time.time(),
}

def recompute_totals():
    subtotal = sum(float(it["price"]) * int(it["qty"]) for it in cart_state["items"])
    cart_state["subtotal"] = float(subtotal)
    cart_state["total"] = max(0.0, float(subtotal) - float(cart_state.get("discount", 0.0)))
    cart_state["updated_at"] = time.time()

def broadcast_cart():
    with cart_lock:
        recompute_totals()
        socketio.emit("cart:update", cart_state)

def _deny_if_scale_pending() -> bool:
    return bool(cart_state.get("scale_pending"))

# ---------------- 秤重執行緒 ----------------
class SerialScale:
    """
    request(base_name) 後：
      - pending；穩定後依 split_g 判斷 small/large
      - 自動加入購物車並廣播
    """
    def __init__(self, port=None, baud=9600, debug=True):
        self.port = port or os.environ.get("SCALE_PORT", "COM3")
        self.baud = int(os.environ.get("SCALE_BAUD", baud))
        self.debug = debug

        self.ser = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

        self.pending = False
        self.base_name: Optional[str] = None
        self.rule: Optional[Dict[str, Any]] = None
        self.split_g: Optional[float] = None

        # 讓反應快：~1 秒內穩定
        self.values = deque(maxlen=5)
        self.use_last_k = 5
        self.tol_g = 2.0
        self.min_weight_g = 5.0
        self.stable_need = 1
        self.min_request_ms = 300

        self._stable_pass = 0
        self._request_ts = 0.0
        self._last_emit_ts = 0.0
        self._last_w: Optional[float] = None

        self.thread.start()

    def request(self, base_name: str):
        if base_name not in VARIANTS:
            raise RuntimeError("no_variant_rule")
        self.rule = VARIANTS[base_name]
        self.base_name = base_name
        self.split_g = float(self.rule["split_g"])
        self.values.clear()
        self._stable_pass = 0
        self._request_ts = time.time()
        self._last_w = None
        self.pending = True
        try:
            if self.ser:
                self.ser.reset_input_buffer()
        except:
            pass
        self._emit_status()

    def cancel(self):
        self.pending = False
        self.base_name = None
        self.rule = None
        self.split_g = None
        self.values.clear()
        self._stable_pass = 0
        self._emit_status()

    def status(self) -> Dict[str, Any]:
        w = self._last_w
        suggest = None
        will_add = None
        if self.pending and self.rule and (w is not None) and (self.split_g is not None):
            suggest = "large" if w >= self.split_g else "small"
            will_add = self.rule["large"] if suggest == "large" else self.rule["small"]
        return {
            "pending": self.pending,
            "base_name": self.base_name,
            "weight": w,          # 前端用 st.weight
            "suggest": suggest,
            "will_add": will_add,
            "split_g": self.split_g,
        }

    def _loop(self):
        try:
            import serial
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"[Scale] Open {self.port} @ {self.baud}")
        except Exception as e:
            print(f"[Scale] 連接失敗：{e}")
            self.ser = None

        while self.running:
            try:
                if self.ser is None:
                    time.sleep(0.4)
                    continue

                line = self.ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue

                # 解析重量
                num = None
                m = re.search(r"(?:Wt|Weight|重量)\s*[:=]\s*(-?\d+(?:\.\d+)?)", line, re.I)
                if m:
                    num = m.group(1)
                else:
                    nums = re.findall(r"-?\d+(?:\.\d+)?", line)
                    if nums:
                        num = nums[-1]
                if num is None:
                    continue

                g = float(num)
                self._last_w = g
                if self.pending and self.debug:
                    print(f"[Scale][raw] {line}")

                now = time.time()
                if self.pending and now - self._last_emit_ts >= 0.1:
                    self._last_emit_ts = now
                    self._emit_status()

                if not self.pending:
                    continue
                self.values.append(g)
                if len(self.values) < min(self.use_last_k, self.values.maxlen):
                    continue

                last_k = list(self.values)[-self.use_last_k:]
                stable = (max(last_k) - min(last_k) <= self.tol_g) and \
                         (abs(sum(last_k)/len(last_k)) >= self.min_weight_g)
                self._stable_pass = (self._stable_pass + 1) if stable else 0

                if self._stable_pass >= self.stable_need and (now - self._request_ts) * 1000 >= self.min_request_ms:
                    self._finalize(sum(last_k) / len(last_k))

            except Exception as e:
                if self.debug:
                    print(f"[Scale] loop error: {e}")
                time.sleep(0.05)

    def _emit_status(self):
        socketio.emit("scale:status", self.status())

    def _finalize(self, weight_g: float):
        if not (self.pending and self.rule and self.base_name and self.split_g is not None):
            return
        variant = self.rule["large"] if weight_g >= self.split_g else self.rule["small"]
        price = PRODUCTS.get(variant)
        if price is None:
            print(f"[Scale] 找不到價格：{variant}（請在 products.csv 加上）")
            with cart_lock:
                cart_state["scale_pending"] = False
                cart_state["scale_base"] = None
            socketio.emit("scale:error", {"name": variant, "msg": "price_missing"})
            self.cancel()
            return

        with cart_lock:
            merged = False
            for it in cart_state["items"]:
                if it["name"] == variant and float(it["price"]) == float(price):
                    it["qty"] += 1
                    it["subtotal"] = float(it["price"]) * it["qty"]
                    merged = True
                    break
            if not merged:
                cart_state["items"].append(
                    {"name": variant, "price": float(price), "qty": 1, "subtotal": float(price)}
                )

            recompute_totals()
            cart_state["scale_pending"] = False
            cart_state["scale_base"] = None

        broadcast_cart()
        socketio.emit("scale:finalized", {"name": variant, "weight_g": float(weight_g)})
        self.cancel()

# 啟動秤重執行緒
scale = SerialScale(
    port=os.environ.get("SCALE_PORT", "COM3"),
    baud=int(os.environ.get("SCALE_BAUD", "9600")),
    debug=True,
)

# ---------------- 字型（避免中文亂碼；無浮水印版本） ----------------
def _setup_cjk_fonts():
    """
    1) 若有 fonts/NotoSansTC-Regular.ttf 與 NotoSansTC-Bold.ttf -> 直接註冊/嵌入
    2) 否則退回 ReportLab 內建 CID：
       - Regular: MSung-Light（繁中）
       - Bold   : HeiseiKakuGo-W5（若不可用就同 Regular）
    """
    fonts_dir = BASE_DIR / "fonts"
    reg = fonts_dir / "NotoSansTC-Regular.ttf"
    bold = fonts_dir / "NotoSansTC-Bold.ttf"

    try:
        if reg.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("NotoSansTC", str(reg)))
            pdfmetrics.registerFont(TTFont("NotoSansTC-Bold", str(bold)))
            return "NotoSansTC", "NotoSansTC-Bold"
    except Exception:
        pass

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light"))
        regular_name = "MSung-Light"
    except Exception:
        regular_name = "Helvetica"

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
        bold_name = "HeiseiKakuGo-W5"
    except Exception:
        bold_name = regular_name

    return regular_name, bold_name

PDF_FONT_REG, PDF_FONT_BOLD = _setup_cjk_fonts()

# ---------------- 收據產生（僅左上角校徽，無浮水印） ----------------
RECEIPTS_DIR = os.path.join(os.getcwd(), "receipts")
os.makedirs(RECEIPTS_DIR, exist_ok=True)

def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def _txn_id_now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))

def generate_receipt_pdf(txn_id: str, items, total: float) -> str:
    """
    無浮水印；左上角小 LOGO（static/fcu_logo.png），再畫標題與明細
    """
    filename = f"receipt_{txn_id}.pdf"
    filepath = os.path.join(RECEIPTS_DIR, filename)

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4
    margin_l, margin_r = 50, width - 50
    y = height - 60

    # 左上角 LOGO（存在才畫）
    logo_path = BASE_DIR / "static" / "fcu_logo.png"
    title_x = margin_l
    if logo_path.exists():
        try:
            img = ImageReader(str(logo_path))
            iw, ih = img.getSize()
            # 目標高度 24pt，等比縮放
            target_h = 24
            scale = target_h / float(ih) if ih else 1.0
            w_new = iw * scale
            c.drawImage(img, margin_l, y - target_h + 6, width=w_new, height=target_h, mask='auto')
            title_x = margin_l + w_new + 10  # 標題換到 LOGO 右側
        except Exception as e:
            print(f"[PDF] 無法載入 LOGO：{e}")

    # 標題
    c.setFont(PDF_FONT_BOLD, 18)
    c.drawString(title_x, y, "===== 自助結帳 POS 收據 =====")
    y -= 28

    # 交易資訊
    c.setFont(PDF_FONT_REG, 12)
    c.drawString(margin_l, y, f"交易編號：{txn_id}")
    y -= 20
    c.drawString(margin_l, y, f"交易時間：{_fmt_ts(time.time())}")
    y -= 16

    # 分隔線
    c.setLineWidth(0.6)
    c.line(margin_l, y, margin_r, y)
    y -= 24

    # 欄位抬頭
    x_name = margin_l
    x_price = margin_l + 300
    x_qty = margin_l + 380
    x_sub = margin_l + 450

    c.setFont(PDF_FONT_BOLD, 12)
    c.drawString(x_name,  y, "品名")
    c.drawString(x_price, y, "單價")
    c.drawString(x_qty,   y, "數量")
    c.drawString(x_sub,   y, "小計")
    y -= 20

    c.setFont(PDF_FONT_REG, 12)
    for it in items:
        name = str(it["name"])
        qty = int(it["qty"])
        price = float(it["price"])
        sub = float(it["subtotal"])

        # 超出寬度就截斷加 …
        maxw = x_price - x_name - 10
        text_w = pdfmetrics.stringWidth(name, PDF_FONT_REG, 12)
        if text_w > maxw:
            while name and pdfmetrics.stringWidth(name + "…", PDF_FONT_REG, 12) > maxw:
                name = name[:-1]
            name += "…"

        c.drawString(x_name,  y, name)
        c.drawRightString(x_price + 35, y, f"{price:.0f}")
        c.drawRightString(x_qty + 20,   y, f"{qty}")
        c.drawRightString(x_sub + 35,   y, f"{sub:.0f}")
        y -= 18
        if y < 120:
            c.showPage()
            y = height - 60
            c.setFont(PDF_FONT_REG, 12)

    # 分隔線
    y -= 6
    c.line(margin_l, y, margin_r, y)
    y -= 24

    # 總計/付款/找零（目前預設：付款 = 總金額；找零 = 0）
    paid = total
    change = max(0.0, paid - total)

    c.setFont(PDF_FONT_BOLD, 13)
    c.drawString(margin_l, y, f"總金額：{total:.0f}");   y -= 20
    c.drawString(margin_l, y, f"付款金額：{paid:.0f}"); y -= 20
    c.drawString(margin_l, y, f"找零：{change:.0f}");   y -= 26

    c.setFont(PDF_FONT_REG, 12)
    c.drawString(margin_l, y, "感謝您的購買！")

    c.save()
    return filename

# ---------------- 路由 / API ----------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/cart", methods=["GET"])
def api_get_cart():
    with cart_lock:
        recompute_totals()
        return jsonify(cart_state)

@app.route("/api/cart/add", methods=["POST"])
def api_cart_add():
    if _deny_if_scale_pending():
        return jsonify({"ok": False, "error": "scale_pending"}), 409
    d = request.get_json(force=True)
    name = (d.get("name") or "").strip()
    try:
        price = float(d.get("price", 0))
    except:
        price = 0
    try:
        qty = int(d.get("qty", 1))
    except:
        qty = 0
    if not name or price <= 0 or qty <= 0:
        return jsonify({"ok": False, "error": "invalid item"}), 400

    with cart_lock:
        merged = False
        for it in cart_state["items"]:
            if it["name"] == name and float(it["price"]) == float(price):
                it["qty"] += qty
                it["subtotal"] = float(it["price"]) * it["qty"]
                merged = True
                break
        if not merged:
            cart_state["items"].append({"name": name, "price": float(price), "qty": qty, "subtotal": float(price) * qty})
        recompute_totals()
    broadcast_cart()
    return jsonify({"ok": True})

@app.route("/api/cart/clear", methods=["POST"])
def api_cart_clear():
    if _deny_if_scale_pending():
        return jsonify({"ok": False, "error": "scale_pending"}), 409
    with cart_lock:
        cart_state["items"].clear()
        cart_state["discount"] = 0.0
        cart_state["checkout_pending"] = False
        recompute_totals()
    broadcast_cart()
    return jsonify({"ok": True})

# —— 秤重 API —— #
@app.route("/api/scale/request", methods=["POST"])
def api_scale_request():
    d = request.get_json(force=True)
    base_name = (d.get("base_name") or "").strip()
    if base_name not in VARIANTS:
        return jsonify({"ok": False, "error": "no_variant_rule"}), 400
    with cart_lock:
        cart_state["scale_pending"] = True
        cart_state["scale_base"] = base_name
        cart_state["updated_at"] = time.time()
    try:
        scale.request(base_name)
    except Exception as e:
        with cart_lock:
            cart_state["scale_pending"] = False
            cart_state["scale_base"] = None
        return jsonify({"ok": False, "error": str(e)}), 500
    socketio.emit("scale:status", scale.status())
    return jsonify({"ok": True})

@app.route("/api/scale/cancel", methods=["POST"])
def api_scale_cancel():
    try:
        scale.cancel()
    finally:
        with cart_lock:
            cart_state["scale_pending"] = False
            cart_state["scale_base"] = None
            cart_state["updated_at"] = time.time()
        socketio.emit("scale:status", scale.status())
    return jsonify({"ok": True})

@app.route("/api/scale/status", methods=["GET"])
def api_scale_status():
    return jsonify(scale.status())

# —— 結帳 —— #
@app.route("/api/checkout", methods=["POST"])
def api_checkout():
    if _deny_if_scale_pending():
        return jsonify({"ok": False, "error": "scale_pending"}), 409

    with cart_lock:
        cart_state["checkout_pending"] = True
        total = cart_state["total"]
        items = list(cart_state["items"])
        cart_state["updated_at"] = time.time()

    txn_id = _txn_id_now()

    filename = generate_receipt_pdf(txn_id, items, total)
    url = f"{PUBLIC_BASE_URL}/receipts/{filename}"

    socketio.emit("checkout:confirmed", {"total": total})
    socketio.emit("receipt:ready", {"txn_id": txn_id, "url": url})

    # 1 分鐘後自動清空購物車（可重新結帳）
    def reset_after_delay():
        time.sleep(60)
        with cart_lock:
            cart_state["items"].clear()
            cart_state["discount"] = 0.0
            cart_state["checkout_pending"] = False
            recompute_totals()
        broadcast_cart()
    threading.Thread(target=reset_after_delay, daemon=True).start()

    return jsonify({"ok": True, "total": total, "receipt_url": url})

# —— 收據檔案服務 —— #
@app.route("/receipts/<path:filename>")
def serve_receipt(filename):
    return send_from_directory(RECEIPTS_DIR, filename, as_attachment=False)

# （可選）Server 產生 QR 圖檔的 API；目前前端用 qrcodejs
@app.route("/api/qr")
def api_qr():
    data = request.args.get("data", "").strip()
    if not data:
        return jsonify({"error": "missing data"}), 400
    img = qrcode.make(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

# ---------------- 自動逾時清空（保險機制） ----------------
def gc_expired_cart(timeout=300):
    while True:
        try:
            if time.time() - cart_state.get("updated_at", 0) > timeout and cart_state["items"]:
                with cart_lock:
                    cart_state["items"].clear()
                    cart_state["discount"] = 0.0
                    cart_state["checkout_pending"] = False
                    cart_state["scale_pending"] = False
                    cart_state["scale_base"] = None
                    recompute_totals()
                broadcast_cart()
        except:
            pass
        time.sleep(5)

# ---------------- 入口 ----------------
if __name__ == "__main__":
    threading.Thread(target=gc_expired_cart, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
