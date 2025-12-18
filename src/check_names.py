from ultralytics import YOLO
m = YOLO("best.pt")
print(m.names)  # 會印出 {類別編號: '類別名稱', ...}
