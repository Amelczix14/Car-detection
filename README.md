# License Plate Access Control System

This project is an intelligent access control system based on real-time vehicle license plate recognition. The system uses object detection and OCR techniques to identify cars, extract license plates, recognize vehicle color and brand, and determine access rights based on a pre-defined whitelist.

## \:car: Key Features

* **Live Camera Feed**: Real-time video stream from a camera (iPhone shared via macOS without additional drivers).
* **Scan History**: Log of detected license plates with access status, plate number, vehicle brand, color, and timestamp.
* **Access Management**: Admin panel to add or remove vehicles from the allowed list.
* **Demo Mode**: Simulated detection using sample videos when a live camera is unavailable.

## 📂 Project Structure

```
├── app/
│   └── Core application logic and web interface
├── models/
│   ├── my_model/               # YOLO model for license plate detection
│   ├── car_model/              # YOLO model for front-view car detection
│   └── brand_model/            # YOLO model for car brand recognition
├── detection/
│   ├── plate_detection.py      # Detects and crops license plate from frame
│   ├── ocr_detection.py        # Processes the plate image (denoising, binarization, OCR via Tesseract)
│   ├── car_detection.py        # Detects full vehicle from front if a plate is found
│   ├── brand_detection.py      # Detects vehicle brand using YOLO model
│   └── color_detection.py      # Detects car color using HSV color space
├── docker-compose.yml          # Docker configuration for the database
```

## 🛠️ Technologies Used

* Python
* YOLOv11
* OpenCV
* Tesseract OCR
* Flask
* PostgreSQL (via Docker)

---

## 🚀 Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Amelczix14/Car-detection.git
   cd Car-detection
   ```

2. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the PostgreSQL database with Docker**:

   ```bash
   docker compose up -d
   ```

4. **Run the web application**:

   ```bash
   python app/app.py
   ```

5. **Open in browser**:
   Go to [http://localhost:5000](http://localhost:5000)

---

## 🧪 Testing with Your Own Videos

You can test the system using your own `.mp4` videos by placing them in the `app/videos/` and altering existing paths in `app/index.html`.

> ⚠️ Keep in mind:
>
> * **Not all car brands and colors are currently recognized** since models are trained on a limited set.
> * **Color detection** is based on HSV space and may not be reliable under varying lighting conditions (e.g., shadows, night scenes, or artificial lights).

---

## 🔮 Planned Improvements

* Extend the brand recognition model to support more car makes.
* Improve color classification by handling more color variations and improving robustness to lighting changes.
* Optionally integrate license plate country or region detection.

---
