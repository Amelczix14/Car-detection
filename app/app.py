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
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String)  # GRANTED or DENIED

# Flask
app = Flask(__name__)

cap = cv2.VideoCapture(camera_number)
plate_detector = PlateDetection('../models/my_model/my_model.pt')
car_detector = CarDetection('../models/car_model/car_model.pt')
color_detector = ColorDetection()
brand_detector = BrandDetection('../models/brand_model/best.pt')

def generate_frames(video_path=None):
    cap = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(camera_number)

    last_granted_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        plate_img = plate_detector._process_frame(frame)
        plate_number = "NO_PLATE"
        status = "PROCESSING"
        car_color = None
        car_brand = None

        if plate_img is not None and (time.time() - last_granted_time > 30):
            reads = []
            for _ in range(3):
                plate_text, conf = read_licence_plate(plate_img)
                if plate_text or len(plate_text)>4:
                    reads.append(plate_text)
            if reads:
                plate_number = max(set(reads), key=reads.count)

            session = Session()
            found = session.query(Plate).filter_by(plate_number=plate_number).first()
            status = 'GRANTED' if found else 'DENIED'

            if plate_number:
                car_results = car_detector.model(frame, verbose=False)
                for det in car_results[0].boxes:
                    if det.conf > car_detector.min_thresh:
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        car_crop = frame[y1:y2, x1:x2]
                        car_color = color_detector.classify_color(car_crop)

                        # === Marka samochodu ===
                        car_brand, brand_crop, brand_coords = brand_detector.detect_brand(frame)
                        if brand_coords:
                            bx1, by1, bx2, by2 = brand_coords
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)  # Czerwony prostokąt
                            cv2.putText(frame, "brand", (bx1, by1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            session.add(Log(plate_number=plate_number, status=status))
            session.commit()
            session.close()

            if status == "GRANTED":
                last_granted_time = time.time()

            current_plate_result["plate"] = plate_number
            current_plate_result["status"] = status
            current_plate_result["color"] = car_color
            current_plate_result["brand"] = car_brand
            current_plate_result["timestamp"] = time.time()

        # bounding box dla rejestracji
        if plate_detector.last_bbox is not None:
            x1, y1, x2, y2 = plate_detector.last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if plate_number != "NO_PLATE":
                cv2.putText(frame, "plate", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
            'timestamp': log.timestamp.isoformat(),
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
    current_plate_result.update({
        "plate": None,
        "status": None,
        "color": None,
        "brand": None,
        "timestamp": None
    })
    plate_detector.last_bbox = None

current_plate_result = {"plate": None, "status": None, "color": None, "brand": None, "timestamp": None}

@app.route('/latest_detection')
def latest_detection():
    return jsonify(current_plate_result)

@app.route('/current_video')
def get_current_video():
    return jsonify({'filename': CURRENT_VIDEO})


if __name__ == '__main__':
    if not os.path.exists("static/logs"):
        os.makedirs("static/logs")
    app.run(debug=True)
