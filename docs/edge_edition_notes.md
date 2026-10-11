# DrishtiDrive AI — Raspberry Pi Edge Edition Architecture Notes

**Document:** `edge_edition_notes.md`  
**Version:** 1.0  
**Target:** Future Raspberry Pi 4B / 5 Embedded Deployment  

---

## 1. Hardware Architecture & Interfaces

The Raspberry Pi Edge Edition packages DrishtiDrive AI into an autonomous dashboard-mounted IoT device:

- **Compute Unit:** Raspberry Pi 4B (4GB/8GB) or Raspberry Pi 5.
- **Vision Sensor:** Raspberry Pi Camera Module 3 (Wide-angle) or USB UVC infrared camera (for night-vision cabin monitoring).
- **Audio Output:** 3.5mm audio jack or I2S DAC amplifier (e.g., MAX98357A) connected to a 3W 4Ω cabin speaker.
- **Visual Indicators:** High-intensity RGB LED bar (Green: Alert, Yellow: Stage 1, Red: Stage 2/3) or 0.96" OLED I2C telemetry display.
- **Physical Controls:** Driver acknowledge/snooze tactile push button mounted on the casing.

---

## 2. Power Supply & Electrical Isolation

> [!WARNING]
> **Automotive Electrical Isolation Safety Rule:**  
> The device must be powered exclusively from the vehicle's standard 12V auxiliary / cigarette lighter socket via an isolated step-down buck converter (12V/24V to 5V 3A, USB-C PD).  
> The device **must never** tap into the vehicle CAN bus, ECU, horn relay, steering, or brake lines.

---

## 3. Lightweight Edge Optimization

1. **Model Format Conversion:**
   - Convert Keras model (`mobilenetv2.keras`) to **TensorFlow Lite (`.tflite`)** with FP16 or INT8 post-training quantization.
   - Alternatively, export to **ONNX Runtime** utilizing ARM NEON execution providers.
2. **Performance Target:**
   - Inference Latency: < 35ms per eye crop on Raspberry Pi 4 CPU.
   - Frame Rate: Stable 15–20 FPS webcam pipeline.
   - Thermal Budget: Active cooling fan maintaining SoC temperature < 65°C under continuous cabin heat.

---

## 4. Benchmarking & Deployment Checklist

- [ ] Flash Raspberry Pi OS Lite (64-bit).
- [ ] Enable Camera interface (`raspi-config`).
- [ ] Configure `systemd` service (`drishtidrive.service`) for automatic headless boot.
- [ ] Validate thermal throttling under 2-hour sustained load test.
- [ ] Test acoustic audibility against typical 70–80 dB truck cabin engine noise.
