# pos.py
import os, csv, threading, time, random, socket
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import requests

# ====== 自動偵測區網 IP（POS 與 server 溝通用）======
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 不需真的連線，這裡只是觸發系統選路由以取本機區網 IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

SERVER = f"http://{get_local_ip()}:5000"
print("[POS] 使用伺服器位址:", SERVER)

# ====== 與 server 同步 ======
def sync_add(name, price, qty):
    try:
        requests.post(
            f"{SERVER}/api/cart/add",
            json={"name": name, "price": float(price), "qty": int(qty)},
            timeout=1.5,
        )
    except Exception as e:
        print("同步新增失敗：", e)

def sync_clear():
    try:
        requests.post(f"{SERVER}/api/cart/clear", timeout=1.5)
    except Exception as e:
        print("同步清空失敗：", e)

def sync_checkout():
    try:
        requests.post(f"{SERVER}/api/checkout", timeout=1.5)
    except Exception as e:
        print("同步結帳通知失敗：", e)

def sync_checkout_ack():
    try:
        requests.post(f"{SERVER}/api/checkout/ack", timeout=1.5)
    except Exception as e:
        print("結帳 ACK 失敗：", e)

def sync_receipt_ready(txn_id):
    try:
        requests.post(
            f"{SERVER}/api/receipt_ready",
            json={"txn_id": txn_id},
            timeout=1.5
        )
    except Exception as e:
        print("通知收據失敗：", e)

# ====== PDF 收據 ======
def _ensure_reportlab():
    try:
        import reportlab  # noqa
        return True
    except Exception:
        messagebox.showerror(
            "缺少套件",
            "需要安裝 reportlab 才能輸出 PDF 收據。\n\n請在終端機執行： pip install reportlab"
        )
        return False

def _register_cjk_font():
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        candidates = [
            ("MSJH", r"C:\Windows\Fonts\msjh.ttc"),
            ("MSYH", r"C:\Windows\Fonts\msyh.ttc"),
            ("SimHei", r"C:\Windows\Fonts\simhei.ttf"),
            ("MingLiu", r"C:\Windows\Fonts\mingliu.ttc"),
        ]
        for name, path in candidates:
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont(name, path))
                return name
    except Exception:
        pass
    return "Helvetica"

# ====== 常用商品 ======
QUICK_ITEMS = [
    ("香蕉", 25), ("蘋果", 30), ("牛奶", 45), ("麵包", 35),
    ("礦泉水", 20), ("咖啡", 55), ("餅乾", 40), ("泡麵", 35), ("西瓜", 100),
]

class POSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("自助結帳 POS")
        self.root.geometry("820x640")

        # 狀態
        self.cart = {}          # {name: {"price": float, "qty": int}}
        self.price_map = {}
        self.price_src_path = None
        self._processing_remote_checkout = False

        # ====== 菜單 ======
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="載入價目表 (CSV)", command=self.load_price_csv)
        menubar.add_cascade(label="檔案", menu=filemenu)
        root.config(menu=menubar)

        # ====== 上方輸入區 ======
        top = tk.Frame(root); top.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(top, text="品名").grid(row=0, column=0, padx=4, sticky="w")
        self.name_var = tk.StringVar()
        tk.Entry(top, textvariable=self.name_var, width=20).grid(row=0, column=1, sticky="w")
        tk.Label(top, text="單價").grid(row=0, column=2, padx=8, sticky="w")
        self.price_var = tk.StringVar()
        tk.Entry(top, textvariable=self.price_var, width=10).grid(row=0, column=3, sticky="w")
        tk.Button(top, text="帶出單價", width=10, command=self.fill_price_from_map).grid(row=0, column=4, padx=8)
        tk.Button(top, text="加入", width=10, command=self.on_add).grid(row=0, column=5)

        self.src_label = tk.Label(root, text="（尚未載入價目表）", fg="#666")
        self.src_label.pack(anchor="w", padx=12)

        # ====== 快速按鈕 ======
        quick = tk.LabelFrame(root, text="常用商品")
        quick.pack(fill=tk.X, padx=10, pady=(6,8))
        for idx, (name, price) in enumerate(QUICK_ITEMS):
            tk.Button(
                quick, text=f"{name}\n${price}", width=10, height=2,
                command=lambda n=name, p=price: self.quick_add(n, p)
            ).grid(row=0, column=idx, padx=4, pady=4)

        # ====== 購物車表格 ======
        columns = ("name", "price", "qty", "subtotal")
        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=12)
        self.tree.heading("name", text="品名")
        self.tree.heading("price", text="單價")
        self.tree.heading("qty", text="數量")
        self.tree.heading("subtotal", text="小計")
        self.tree.column("name", width=320)
        self.tree.column("price", width=80, anchor="e")
        self.tree.column("qty", width=60, anchor="center")
        self.tree.column("subtotal", width=100, anchor="e")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10)

        # ====== 功能按鈕 ======
        btns = tk.Frame(root); btns.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(btns, text="＋1", width=8, command=self.on_inc).pack(side=tk.LEFT)
        tk.Button(btns, text="−1", width=8, command=self.on_dec).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="刪除", width=10, command=self.on_delete).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="清空", width=8, command=self.on_clear).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="模擬辨識", width=10, command=self.simulate_detect).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="結帳", width=10, command=self.on_checkout).pack(side=tk.RIGHT, padx=6)

        self.total_var = tk.StringVar(value="總價：0")
        tk.Label(root, textvariable=self.total_var, font=("Microsoft JhengHei", 14)).pack(anchor="e", padx=12, pady=6)

        # ====== 背景輪詢顧客端狀態（偵測 checkout_pending 與同步 cart）======
        threading.Thread(target=self._poll_server, daemon=True).start()

    # ====== 載入價目表 ======
    def load_price_csv(self):
        path = filedialog.askopenfilename(title="選擇價目表 CSV", filetypes=[("CSV files", "*.csv")])
        if not path: return
        try:
            price_map = {}
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("name") or "").strip()
                    try:
                        price_map[name.lower()] = float(row.get("price"))
                    except:
                        pass
            self.price_map = price_map
            self.src_label.config(text=f"已載入價目表：{os.path.basename(path)}", fg="#1a7f37")
        except Exception as e:
            messagebox.showerror("讀取失敗", str(e))

    def lookup_price(self, name):
        return self.price_map.get(name.lower(), 0.0)

    def fill_price_from_map(self):
        name = self.name_var.get().strip()
        if not name: return
        price = self.lookup_price(name)
        if price:
            self.price_var.set(str(int(price)))

    # ====== 顧客端 → POS 同步（含待結帳偵測）======
    def _apply_server_cart(self, cart_json):
        new_cart = {
            it["name"]: {"price": float(it["price"]), "qty": int(it["qty"])}
            for it in (cart_json.get("items") or [])
        }
        if new_cart != self.cart:
            self.cart = new_cart
            self.refresh_table()
        if cart_json.get("checkout_pending") and not self._processing_remote_checkout:
            self._processing_remote_checkout = True
            self.root.after(0, self._finalize_remote_checkout)

    def _poll_server(self):
        while True:
            try:
                r = requests.get(f"{SERVER}/api/cart", timeout=1.5)
                if r.ok:
                    data = r.json()
                    self.root.after(0, lambda d=data: self._apply_server_cart(d))
            except:
                pass
            time.sleep(1)

    def _finalize_remote_checkout(self):
        """顧客端按了確認 → 以實付=總額自動完成結帳，產出收據與 QR，清空雙方."""
        total = sum(info["price"] * info["qty"] for info in self.cart.values())
        if total <= 0:
            sync_checkout_ack()
            self._processing_remote_checkout = False
            return
        pay, change = total, 0
        txn_id = self.save_transaction(pay, change)
        self.generate_receipt_txt(txn_id, pay, change)
        pdf_path = self.generate_receipt_pdf(txn_id, pay, change)
        # 通知 server 生成收據 URL → 顧客端會顯示 QR 與連結
        sync_receipt_ready(txn_id)
        # 清 pending 與清空
        sync_checkout_ack()
        sync_clear()
        self.cart.clear()
        self.refresh_table()
        messagebox.showinfo("完成", f"顧客端確認 → 自動結帳\n已輸出收據：{pdf_path}")
        self._processing_remote_checkout = False

    # ====== 購物車操作（POS → 顧客端同步）======
    def quick_add(self, name, price):
        use_price = self.lookup_price(name) or price
        self.cart.setdefault(name, {"price": use_price, "qty": 0})
        self.cart[name]["qty"] += 1
        self.refresh_table()
        sync_add(name, use_price, 1)

    def on_add(self):
        name, price_str = self.name_var.get().strip(), self.price_var.get().strip()
        if not name:
            return
        try:
            price = float(price_str or self.lookup_price(name))
        except:
            messagebox.showwarning("提示", "單價需為數字或請先載入價目表")
            return
        if price <= 0:
            messagebox.showwarning("提示", "單價需大於 0")
            return
        self.cart.setdefault(name, {"price": price, "qty": 0})
        self.cart[name]["qty"] += 1
        self.refresh_table()
        sync_add(name, price, 1)
        self.name_var.set("")
        self.price_var.set("")

    def on_inc(self):
        name = self._get_selected_name()
        if name:
            self.quick_add(name, self.cart[name]["price"])

    def on_dec(self):
        name = self._get_selected_name()
        if not name:
            return
        self.cart[name]["qty"] -= 1
        if self.cart[name]["qty"] <= 0:
            del self.cart[name]
        self.refresh_table()
        # 重新同步（簡單做法：清空後重放）
        sync_clear()
        for n, info in self.cart.items():
            sync_add(n, info["price"], info["qty"])

    def on_delete(self):
        name = self._get_selected_name()
        if name:
            del self.cart[name]
            self.refresh_table()
            sync_clear()
            for n, info in self.cart.items():
                sync_add(n, info["price"], info["qty"])

    def on_clear(self):
        self.cart.clear()
        self.refresh_table()
        sync_clear()

    def simulate_detect(self):
        name, price = random.choice(QUICK_ITEMS)
        self.quick_add(name, price)
        messagebox.showinfo("模擬辨識", f"已偵測到商品：{name} (${price})")

    # ====== 交易與收據 ======
    def save_transaction(self, pay, change):
        os.makedirs("data", exist_ok=True)
        txn_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = sum(info["price"] * info["qty"] for info in self.cart.values())
        path = os.path.join("data", "transactions.csv")
        write_header = not os.path.exists(path)
        rows = [
            {
                "txn_id": txn_id,
                "timestamp": ts,
                "item_name": n,
                "unit_price": int(i["price"]),
                "qty": int(i["qty"]),
                "subtotal": int(i["price"] * i["qty"]),
                "total": int(total),
                "pay": int(pay),
                "change": int(change),
            }
            for n, i in self.cart.items()
        ]
        if rows:
            with open(path, "a", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                if write_header:
                    w.writeheader()
                w.writerows(rows)
        return txn_id

    def generate_receipt_txt(self, txn_id, pay, change):
        os.makedirs("receipts", exist_ok=True)
        path = os.path.join("receipts", f"receipt_{txn_id}.txt")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = sum(i["price"] * i["qty"] for i in self.cart.values())
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("===== 自助結帳 POS 收據 =====\n")
            f.write(f"交易編號：{txn_id}\n交易時間：{ts}\n")
            f.write("--------------------------------\n品名 單價 數量 小計\n")
            for n, i in self.cart.items():
                f.write(f"{n} {int(i['price'])} {i['qty']} {int(i['price']*i['qty'])}\n")
            f.write("--------------------------------\n")
            f.write(f"總金額：{int(total)}\n付款金額：{int(pay)}\n找零：{int(change)}\n感謝您的購買！\n")
        return path

    def generate_receipt_pdf(self, txn_id, pay, change):
        if not _ensure_reportlab():
            return None
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
        font_name = _register_cjk_font()
        os.makedirs("receipts", exist_ok=True)
        path = os.path.join("receipts", f"receipt_{txn_id}.pdf")
        # 簡版一頁（高度夠用），若要自動分頁可再加強
        W, H, margin, line_h = 80*mm, 200*mm, 5*mm, 6*mm
        c = canvas.Canvas(path, pagesize=(W, H))
        c.setFont(font_name, 10)
        y = H - margin

        def writeln(t):
            nonlocal y
            c.drawString(margin, y, t)
            y -= line_h

        writeln("===== 自助結帳 POS 收據 =====")
        writeln(f"交易編號：{txn_id}")
        writeln(f"交易時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        writeln("--------------------------------")
        writeln("品名 單價 數量 小計")
        total = 0
        for n, i in self.cart.items():
            sub = int(i["price"] * i["qty"]); total += sub
            writeln(f"{n} {int(i['price'])} {i['qty']} {sub}")
        writeln("--------------------------------")
        writeln(f"總金額：{int(total)}")
        writeln(f"付款金額：{int(pay)}")
        writeln(f"找零：{int(change)}")
        writeln("感謝您的購買！")
        c.showPage()
        c.save()
        return path

    # ====== 結帳（POS 主動）======
    def on_checkout(self):
        total = sum(i["price"] * i["qty"] for i in self.cart.values())
        if total <= 0:
            messagebox.showinfo("提示", "購物車是空的")
            return
        win = tk.Toplevel(self.root); win.title("結帳"); win.geometry("300x170")
        tk.Label(win, text=f"總金額：{int(total)} 元").pack(pady=10)
        tk.Label(win, text="付款金額").pack()
        pay_var = tk.StringVar(); tk.Entry(win, textvariable=pay_var).pack()

        def confirm():
            try:
                pay = float(pay_var.get())
                if pay < total:
                    messagebox.showwarning("提示", "付款不足")
                    return
                change = pay - total
                txn_id = self.save_transaction(pay, change)
                self.generate_receipt_txt(txn_id, pay, change)
                pdf_path = self.generate_receipt_pdf(txn_id, pay, change)
                # 通知 server 讓顧客端顯示 QR 與下載連結
                sync_receipt_ready(txn_id)
                # 清 pending、告知顧客端已結帳並清空
                sync_checkout_ack()
                sync_checkout()
                sync_clear()
                self.cart.clear()
                self.refresh_table()
                win.destroy()
                messagebox.showinfo("完成", f"付款 {int(pay)} 找零 {int(change)}\n已輸出收據：{pdf_path}")
            except:
                messagebox.showwarning("提示", "輸入錯誤")

        tk.Button(win, text="確認付款", command=confirm).pack(pady=10)

    # ====== 共用 ======
    def _get_selected_name(self):
        sel = self.tree.selection()
        return self.tree.item(sel[0], "values")[0] if sel else None

    def refresh_table(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        total = 0
        for n, i in self.cart.items():
            subtotal = i["price"] * i["qty"]; total += subtotal
            self.tree.insert("", tk.END, values=(n, f"{i['price']}", i["qty"], f"{int(subtotal)}"))
        self.total_var.set(f"總價：{int(total)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = POSApp(root)
    root.mainloop()
