/*
 * HX711 + 1602 I2C LCD (PCF8574)  — 持久化校正到 EEPROM + 平滑濾波
 * 指令：
 *   t        -> 去皮 (空盤時按)
 *   m <克數> -> 設定砝碼重量，例如: m 100
 *   c        -> 自動計算校正係數，並自動存到 EEPROM
 *   w        -> 立刻把目前的 scale/offset 再存一次到 EEPROM
 *   r        -> 顯示原始平均值(raw)與offset
 *   ?        -> 說明
 */

#include <HX711.h>
#include <Wire.h>
#include <LiquidCrystal_PCF8574.h>
#include <EEPROM.h>
#include <math.h>

// ====== LCD 設定 ======
#define LCD_ADDR 0x27                // 若無顯示改 0x3F
LiquidCrystal_PCF8574 lcd(LCD_ADDR);

// ====== HX711 腳位 ======
#define DT_PIN  6
#define SCK_PIN 5
HX711 scale;

// ====== 量測與校正參數 ======
float known_mass_g = 100.0f;     // 砝碼重量（g）
float calibration_factor = 1.0f; // 會從 EEPROM 覆蓋
const float DEAD_BAND = 0.05f;   // 小抖動 ±0.05 g 視為 0

// ====== 濾波用變數（新增） ======
float filtered_g = 0.0f;
bool  filtered_init = false;

// ====== EEPROM 內校正資料格式 ======
struct CalData {
  uint32_t magic;
  float    cal;
  long     offset;
};

const uint32_t CAL_MAGIC = 0xC011CA1u;
const int EEPROM_ADDR = 0;

bool eepromLoad(CalData &out) {
  EEPROM.get(EEPROM_ADDR, out);
  if (out.magic != CAL_MAGIC) return false;
  if (isnan(out.cal) || isinf(out.cal)) return false;
  return true;
}

void eepromSave(float cal, long offset) {
  CalData d;
  d.magic  = CAL_MAGIC;
  d.cal    = cal;
  d.offset = offset;
  EEPROM.put(EEPROM_ADDR, d);
}

// ====== 輔助函式 ======
void printHelp(){
  Serial.println(F("\n指令："));
  Serial.println(F("  t        -> 去皮 (空盤時按)"));
  Serial.println(F("  m <克數> -> 設定砝碼重量，例如: m 100"));
  Serial.println(F("  c        -> 自動計算校正係數（砝碼已放上時按）[自動存檔]"));
  Serial.println(F("  w        -> 將目前 scale/offset 寫入 EEPROM"));
  Serial.println(F("  r        -> 顯示原始平均值(raw)與offset"));
  Serial.println(F("  ?        -> 說明"));
}

void lcdSplash(){
  lcd.clear();
  lcd.setCursor(0,0); lcd.print("HX711 Scale");
  lcd.setCursor(0,1); lcd.print("DT=6  SCK=5");
  delay(1200);
}

void lcdShowMsg(const char* msg){
  lcd.setCursor(0,1);
  lcd.print("                ");
  lcd.setCursor(0,1);
  lcd.print(msg);
}

void setup() {
  Serial.begin(9600);

  // ---- LCD 初始化 ----
  lcd.begin(16, 2);
  lcd.setBacklight(255);
  lcdSplash();

  // ---- HX711 初始化 ----
  scale.begin(DT_PIN, SCK_PIN);

  // ---- EEPROM 校正載入 ----
  CalData d;
  if (eepromLoad(d)) {
    calibration_factor = d.cal;
    scale.set_scale(calibration_factor);
    scale.set_offset(d.offset);
    Serial.print(F("載入校正：cal=")); Serial.print(calibration_factor, 6);
    Serial.print(F(", offset=")); Serial.println(d.offset);
    lcdShowMsg("Cal loaded");
  } else {
    scale.set_scale(calibration_factor);
    Serial.println(F("未找到校正 → 請: t -> 放砝碼 -> m 100 -> c"));
    lcdShowMsg("No cal, do 'c'");
  }

  Serial.println(F("\n步驟：空盤 -> t 去皮 -> 放砝碼 -> m 100 -> c 校正"));
  printHelp();

  lcd.clear();
  lcd.setCursor(0,0); lcd.print("Wt: -----.-- g");
}

void loop() {

  // ====== 指令處理 ======
  if (Serial.available()){
    String s = Serial.readStringUntil('\n');
    s.trim();

    if (s == "t"){
      Serial.print(F("去皮中..."));
      lcdShowMsg("Taring...");
      scale.tare(20);
      Serial.println(F("OK"));
      lcdShowMsg("Tare OK");
    }
    else if (s.startsWith("m ")) {
      known_mass_g = s.substring(2).toFloat();
      Serial.print(F("砝碼重量 = "));
      Serial.print(known_mass_g,2);
      Serial.println(F(" g"));
      lcdShowMsg("Mass OK");
    }
    else if (s == "c") {
      Serial.println(F("計算校正係數...保持砝碼穩定"));
      lcdShowMsg("Calibrating...");
      delay(1200);

      long offset   = scale.get_offset();
      long rawNow   = scale.read_average(10);
      long rawDelta = rawNow - offset;

      if (known_mass_g <= 0.0f) {
        Serial.println(F("請先用 'm <克數>' 設定砝碼"));
        lcdShowMsg("Set m first!");
      } else {
        calibration_factor = (float)rawDelta / known_mass_g;
        scale.set_scale(calibration_factor);

        Serial.print(F("rawDelta=")); Serial.print(rawDelta);
        Serial.print(F(", cal=")); Serial.println(calibration_factor, 6);

        eepromSave(calibration_factor, scale.get_offset());
        Serial.println(F("已保存到 EEPROM"));
        lcdShowMsg("Cal saved");
        delay(800);
      }
    }
    else if (s == "w") {
      eepromSave(scale.get_scale(), scale.get_offset());
      Serial.println(F("已寫入 EEPROM"));
      lcdShowMsg("Saved");
    }
    else if (s == "r") {
      long raw = scale.read_average(10);
      Serial.print(F("raw="));    Serial.print(raw);
      Serial.print(F(", offset=")); Serial.println(scale.get_offset());
      lcdShowMsg("Show raw");
    }
    else if (s == "?") {
      printHelp();
      lcdShowMsg("Help");
    }
  }


  // ====== ★★★ 連續顯示克數（雙濾波 + 歸零） ★★★ ======

  // 第 1 層：HX711 10筆平均
  float g_raw = scale.get_units(3);

  // 第 2 層：指數平均（低通濾波）
  const float alpha = 0.4f;  // 0.05~0.2 可調
  if (!filtered_init) {
    filtered_g = g_raw;
    filtered_init = true;
  } else {
    filtered_g = alpha * g_raw + (1.0f - alpha) * filtered_g;
  }

  float g = filtered_g;

  // 第 3 層：微小抖動視為 0
  if (fabs(g) < DEAD_BAND) g = 0.0f;

  // Serial 顯示
  Serial.print("重量: ");
  Serial.print(g, 2);
  Serial.println(" g");

  // LCD 顯示
  lcd.setCursor(0,0);
  lcd.print("                ");
  lcd.setCursor(0,0);
  lcd.print("Wt: ");
  lcd.print(g, 2);
  lcd.print(" g");

  delay(50);
}
