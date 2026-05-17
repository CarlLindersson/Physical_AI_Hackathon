/*
 * Atech UI Bridge for QArm Mini Recycling Robot
 *
 * Atech handles:
 * - Button: start/stop sorting
 * - Potentiometer: arm speed
 * - Display: material + confidence
 * - Speaker: audio feedback tones
 *
 * Laptop handles:
 * - YOLO
 * - QArm Mini movement
 * - final sorting decision
 */

#include <Arduino.h>
#include <ArduinoJson.h>

#include "modules/connectivity/serial_templates.h"
#include "modules/display/st7735_tft.h"
#include "modules/audio/speaker.h"
#include "modules/input/button.h"

// If Atech has a potentiometer module header, use that.
// If not, replace this with the correct generated include.
#include "modules/input/potentiometer.h"

// ================= SERIAL =================
AtechSerial bridge(115200);

// ================= MODULES =================
// IMPORTANT:
// Keep these constructor values from the Atech generated code/editor
// for YOUR exact ports.

// Color display in ports 9 & 10
ST7735_TFT display_1(2, 41, 1, 40);

// Speaker in ports 3 & 4
Speaker speaker_1(4, 5, 8);

// Button in port 11
ButtonModule start_button(16, true);

// Potentiometer in ports 13 & 14
// CHANGE THIS LINE if Atech generated a different constructor.
Potentiometer pot_1(39);


// ================= STATE =================
bool sortingEnabled = false;

float armSpeed = 0.5;
float lastSentSpeed = -1.0;

String currentMaterial = "None";
float currentConfidence = 0.0;
bool currentUncertain = false;

unsigned long lastControlSend = 0;
unsigned long lastDisplayUpdate = 0;

String lastSpokenMaterial = "";


// ================= HELPERS =================

void sendControlState() {
  StaticJsonDocument<160> doc;

  doc["type"] = "control";
  doc["running"] = sortingEnabled;
  doc["speed"] = armSpeed;

  serializeJson(doc, Serial);
  Serial.println();
}

void drawDisplay() {
  display_1.fillScreen(ST7735_BLACK);

  display_1.setTextSize(1);
  display_1.setTextColor(ST7735_WHITE);
  display_1.setCursor(5, 5);
  display_1.print("RecycleBot");

  display_1.drawLine(0, 17, 159, 17, ST7735_WHITE);

  display_1.setCursor(5, 24);
  display_1.setTextColor(ST7735_CYAN);
  display_1.print("Material:");

  display_1.setCursor(75, 24);

  if (currentUncertain) {
    display_1.setTextColor(ST7735_YELLOW);
  } else {
    display_1.setTextColor(ST7735_GREEN);
  }

  display_1.print(currentMaterial);

  display_1.setCursor(5, 40);
  display_1.setTextColor(ST7735_CYAN);
  display_1.print("Conf:");

  display_1.setCursor(75, 40);
  display_1.setTextColor(currentUncertain ? ST7735_YELLOW : ST7735_GREEN);
  display_1.print((int)(currentConfidence * 100));
  display_1.print("%");

  display_1.setCursor(5, 56);
  display_1.setTextColor(ST7735_CYAN);
  display_1.print("Speed:");

  display_1.setCursor(75, 56);
  display_1.setTextColor(ST7735_WHITE);
  display_1.print((int)(armSpeed * 100));
  display_1.print("%");

  display_1.setCursor(5, 70);
  if (sortingEnabled) {
    display_1.setTextColor(ST7735_GREEN);
    display_1.print("SORTING ENABLED");
  } else {
    display_1.setTextColor(ST7735_RED);
    display_1.print("SORTING STOPPED");
  }
}

void playMaterialTone(String material, bool uncertain) {
  if (uncertain) {
    speaker_1.playTone(220, 250);
    return;
  }

  if (material == "plastic") {
    speaker_1.playTone(700, 180);
  } else if (material == "paper") {
    speaker_1.playTone(500, 180);
  } else if (material == "metal") {
    speaker_1.playTone(900, 180);
  } else if (material == "general") {
    speaker_1.playTone(350, 180);
  } else {
    speaker_1.playTone(300, 120);
  }
}

void handleDetectionMessage(JsonDocument& doc) {
  currentMaterial = doc["material"] | "unknown";
  currentConfidence = doc["confidence"] | 0.0;
  currentUncertain = doc["uncertain"] | false;

  drawDisplay();

  if (currentUncertain) {
    speaker_1.playTone(220, 250);
    lastSpokenMaterial = "uncertain";
  } else if (currentMaterial != lastSpokenMaterial) {
    playMaterialTone(currentMaterial, false);
    lastSpokenMaterial = currentMaterial;
  }
}

void readSerialMessages() {
  if (!Serial.available()) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.length() == 0) {
    return;
  }

  StaticJsonDocument<256> doc;
  DeserializationError error = deserializeJson(doc, line);

  if (error) {
    return;
  }

  const char* type = doc["type"];

  if (!type) {
    return;
  }

  if (strcmp(type, "detection") == 0) {
    handleDetectionMessage(doc);
  }
}


// ================= SETUP =================

void setup() {
  Serial.begin(115200);

  bridge.connect();

  display_1.begin();
  delay(150);

  speaker_1.begin();
  speaker_1.setVolume(0.7);

  start_button.begin();

  pot_1.begin();

  drawDisplay();

  speaker_1.playTone(600, 150);
}


// ================= LOOP =================

void loop() {
  start_button.update();

  if (start_button.wasPressed()) {
    sortingEnabled = !sortingEnabled;

    if (sortingEnabled) {
      speaker_1.playTone(800, 120);
    } else {
      speaker_1.playTone(250, 180);
    }

    sendControlState();
    drawDisplay();
  }

  // Read potentiometer as 0.0 to 1.0
  int rawPot = pot_1.read();
  armSpeed = constrain(rawPot / 4095.0, 0.1, 1.0);

  if (abs(armSpeed - lastSentSpeed) > 0.03 && millis() - lastControlSend > 150) {
    lastSentSpeed = armSpeed;
    lastControlSend = millis();
    sendControlState();
    drawDisplay();
  }

  readSerialMessages();

  delay(10);
}