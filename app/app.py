from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from detection.plate_detection import PlateDetection
from detection.ocr_detection import read_licence_plate
from detection.car_detection import CarDetection
from detection.color_detection import ColorDetection
from detection.brand_detection import BrandDetection
import datetime
import time
import cv2
import os
from datetime import timedelta
from zoneinfo import ZoneInfo


camera_number = 1 # tutaj wpisać odpowiedni numer kamery
rotate = False # zmienna odpowiedzialna za obracanie filmików
CURRENT_VIDEO = "amcia3.mp4" # aktualnie wyświetlany filmik

# Baza danych
DATABASE_URL = 'postgresql://postgres:secret@localhost:5432/access_control'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Modele
class Plate(Base):
    __tablename__ = 'plates'
    id = Column(Integer, primary_key=True)
    plate_number = Column(String, unique=True)

class Log(Base):
    __tablename__ = 'logs'
    id = Column(Integer, primary_key=True)
    plate_number = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(ZoneInfo("Europe/Warsaw")))
    status = Column(String)  # GRANTED or DENIED

    # color = Column(String, nullable=True)  

# Flask
app = Flask(__name__)

cap = cv2.VideoCapture(camera_number)
plate_detector = PlateDetection('../models/my_model/my_model.pt')
car_detector = CarDetection('../models/car_model/car_model.pt')
color_detector = ColorDetection()
brand_detector = BrandDetection('../models/brand_model/best.pt')

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None, None

    # detekcja tablicy
    plate_img = plate_detector._process_frame(frame)
    plate_number = None
    if plate_img is not None:
        reads = []
        for _ in range(3):
            plate_text, conf = read_licence_plate(plate_img)
            if plate_text and len(plate_text) > 4:
                reads.append(plate_text)
        if reads:
            plate_number = max(set(reads), key=reads.count)

    # --- POPRAWKA: sprawdzamy bazę ---
    session = Session()
    found = None
    if plate_number:
        found = session.query(Plate).filter_by(plate_number=plate_number).first()
    session.close()

    status = "GRANTED" if found else "DENIED"

    # detekcja samochodu i koloru
    car_color = None
    car_brand = None
    car_results = car_detector.model(frame, verbose=False)

    for det in car_results[0].boxes:
        if det.conf > car_detector.min_thresh:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            car_crop = frame[y1:y2, x1:x2]
            car_color = color_detector.classify_color(car_crop)

            # marka
            car_brand, brand_crop, brand_coords = brand_detector.detect_brand(frame)
            if brand_coords:
                bx1, by1, bx2, by2 = brand_coords
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(frame, "brand", (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # bounding box tablicy
    if plate_detector.last_bbox is not None:
        x1, y1, x2, y2 = plate_detector.last_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, {
        "plate": plate_number,
        "status": status,
        "color": car_color,
        "brand": car_brand,
        "timestamp": datetime.datetime.now(ZoneInfo("Europe/Warsaw")).isoformat()
    }

# Globalny bufor i stan pauzy
plate_detection_buffer = []  # lista odczytów (True/False) w aktualnym oknie 10s
DETECTION_WINDOW = 5  # czas odczytu w sekundach
pause_until = 0  # timestamp do kiedy pauza po GRANTED
current_plate_window_start = None  # timestamp rozpoczęcia aktualnego okna
last_denied_plate = None  # numer tablicy odrzuconej, aby wyświetlać DENIED do kolejnego odczytu

def generate_frames(video_path=None):
    global plate_detection_buffer, pause_until, current_plate_window_start, last_denied_plate

    cap = cv2.VideoCapture(video_path if video_path else camera_number)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # domyślnie 25 fps jeśli brak info
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now_ts = time.time()

            # Pauza po GRANTED – tylko wyświetlamy status
            if now_ts < pause_until:
                if plate_detector.last_bbox is not None:
                    x1, y1, x2, y2 = plate_detector.last_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if current_plate_result["plate"]:
                        cv2.putText(frame, current_plate_result["plate"], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                _, buffer = cv2.imencode('.jpg', frame)
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    buffer.tobytes() + b'\r\n'
                )
                time.sleep(1 / fps)
                continue

            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            plate_img = plate_detector._process_frame(frame)
            plate_number = "NO_PLATE"
            car_color = None
            car_brand = None

            # Jeśli wykryto tablicę – OCR + detekcja samochodu
            if plate_img is not None:
                plate_text, conf = read_licence_plate(plate_img)
                if plate_text:
                    plate_number = plate_text

                # Detekcja samochodu i koloru/marki tylko po wykryciu tablicy
                car_results = car_detector.model(frame, verbose=False)
                for det in car_results[0].boxes:
                    if det.conf > car_detector.min_thresh:
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        car_crop = frame[y1:y2, x1:x2]
                        car_color = color_detector.classify_color(car_crop)
                        car_brand, brand_crop, brand_coords = brand_detector.detect_brand(frame)
                        if brand_coords and car_brand:
                            bx1, by1, bx2, by2 = brand_coords
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                            cv2.putText(frame, car_brand, (bx1, by1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Aktualizacja current_plate_result – zawsze
            if plate_number == "NO_PLATE" or len(plate_number) <= 4:
                current_plate_result.update({
                    "plate": last_denied_plate,
                    "status": "DENIED" if last_denied_plate else None,
                    "color": car_color,
                    "brand": car_brand,
                    "timestamp": datetime.datetime.now(ZoneInfo("Europe/Warsaw")).isoformat()
                })
                plate_detection_buffer = []
                current_plate_window_start = None
            else:
                # rozpoczęcie nowego okna detekcji
                if current_plate_window_start is None:
                    current_plate_window_start = now_ts
                    plate_detection_buffer = []

                # sprawdzamy w bazie
                session = Session()
                found = session.query(Plate).filter_by(plate_number=plate_number).first()
                session.close()
                plate_detection_buffer.append(True if found else False)

                elapsed = now_ts - current_plate_window_start
                if elapsed >= DETECTION_WINDOW:
                    majority_valid = sum(plate_detection_buffer) > len(plate_detection_buffer) / 2
                    final_status = "GRANTED" if majority_valid else "DENIED"

                    # zapis do logów
                    session = Session()
                    session.add(Log(
                        plate_number=plate_number,
                        status=final_status,
                        timestamp=datetime.datetime.now(ZoneInfo("Europe/Warsaw"))
                    ))
                    session.commit()
                    session.close()

                    current_plate_result.update({
                        "plate": plate_number,
                        "status": final_status,
                        "color": car_color,
                        "brand": car_brand,
                        "timestamp": datetime.datetime.now(ZoneInfo("Europe/Warsaw")).isoformat()
                    })

                    if final_status == "GRANTED":
                        pause_until = now_ts + DETECTION_WINDOW
                        plate_detection_buffer = []
                        current_plate_window_start = None
                        last_denied_plate = None
                    else:
                        last_denied_plate = plate_number
                        plate_detection_buffer = []
                        current_plate_window_start = None

            # Bounding box tablicy i napis
            if plate_detector.last_bbox is not None:
                x1, y1, x2, y2 = plate_detector.last_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if plate_number != "NO_PLATE":
                    cv2.putText(frame, plate_number, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                buffer.tobytes() + b'\r\n'
            )
            time.sleep(1 / fps)
    finally:
        cap.release()



@app.route('/')
def index():
    return render_template('index.html', current_video=CURRENT_VIDEO)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/grant_access', methods=['POST'])
def grant_access():
    plate = request.json.get('plate_number')
    session = Session()
    if not session.query(Plate).filter_by(plate_number=plate).first():
        session.add(Plate(plate_number=plate))
        session.commit()
    session.close()
    return jsonify({'status': 'ok'})

@app.route('/history')
def history():
    session = Session()
    logs = session.query(Log).order_by(Log.timestamp.desc()).all()
    session.close()
    return jsonify([
        {
            'plate_number': log.plate_number,
            # 'color': log.color,
            'timestamp': (log.timestamp + timedelta(hours=2)).isoformat(),
            'status': log.status
        }
        for log in logs
    ])

@app.route('/plates')
def plates():
    session = Session()
    plates = session.query(Plate).all()
    session.close()
    return jsonify([p.plate_number for p in plates])

@app.route('/delete_plate', methods=['POST'])
def delete_plate():
    plate = request.json.get('plate_number')
    session = Session()
    session.query(Plate).filter_by(plate_number=plate).delete()
    session.commit()
    session.close()
    return jsonify({'status': 'deleted'})

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('videos', filename)


@app.route('/video_with_detection')
def video_with_detection():
    video_file = request.args.get('video', CURRENT_VIDEO)
    video_path = os.path.join('videos', video_file)
    if not os.path.exists(video_path):
        return "Video not found", 404

    if request.args.get('reset') == 'true':
        reset_detection_state()

    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

def reset_detection_state():
    global current_plate_result
    global plate_detection_buffer
    global pause_until
    global current_plate_window_start
    global last_denied_plate

    current_plate_result.update({
        "plate": None,
        "status": None,
        "color": None,
        "brand": None,
        "timestamp": None
    })
    plate_detector.last_bbox = None
    plate_detection_buffer = []
    pause_until = 0
    current_plate_window_start = None
    last_denied_plate = None

current_plate_result = {"plate": None, "status": None, "color": None, "brand": None, "timestamp": None}

@app.route('/latest_detection')
def latest_detection():
    return jsonify(current_plate_result)

@app.route('/current_video')
def get_current_video():
    return jsonify({'filename': CURRENT_VIDEO})


from sqlalchemy import text  # dodaj na początku z importami

if __name__ == '__main__':
    if not os.path.exists("static/logs"):
        os.makedirs("static/logs")

    # Czyszczenie historii logów przy każdym uruchomieniu
    session = Session()
    session.execute(text("TRUNCATE TABLE logs RESTART IDENTITY;"))
    session.commit()
    session.close()

    app.run(debug=True)

