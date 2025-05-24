from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from detection.yolo_detection import YOLODetection
from detection.car_detection import CarDetection
from detection.color_detection import ColorDetection
from detection.ocr_detection import read_licence_plate
import datetime
import time
import cv2
import os
from collections import deque, Counter

# Baza danych
DATABASE_URL = 'postgresql://postgres:secret@localhost:5432/access_control'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Historia 
recent_reads = deque(maxlen=5)
last_log_time = 0
cooldown_after_grant = 30 

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

cap = cv2.VideoCapture(0)
plate_detector = YOLODetection('../models/my_model/my_model.pt')
car_detector = CarDetection('../models/car_model/car_model.pt')
color_detector = ColorDetection()

current_plate_result = {"plate": None, "status": None, "timestamp": None}
current_car_color = None

# Dummy detection (do podmiany na YOLO + OCR)
def dummy_detect_plate(frame):
    return "XYZ1234", frame


def generate_frames(video_path=None):
    global  last_log_time, current_car_color

    cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 30 s przerwy po udzieleniu dostępu
        if time.time() - last_log_time < cooldown_after_grant:
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        plate_img = plate_detector._process_frame(frame)

        if plate_img is not None:
            plate_text, conf = read_licence_plate(plate_img)

            if not plate_text or len(plate_text) < 6:
                continue

            recent_reads.append(plate_text)
            most_common, count = Counter(recent_reads).most_common(1)[0]

            if count >= 3:
                plate_number = most_common

                session = Session()
                found = session.query(Plate).filter_by(plate_number=plate_number).first()
                status = 'GRANTED' if found else 'DENIED'
                session.add(Log(plate_number=plate_number, status=status))
                session.commit()
                session.close()

                last_log_time = time.time() 

                current_plate_result["plate"] = plate_number
                current_plate_result["status"] = status
                current_plate_result["timestamp"] = last_log_time

                # TODO: detekcja koloru samochodu
                car_results = car_detector.model(frame, verbose=False)
                for det in car_results[0].boxes:
                    if det.conf > car_detector.min_thresh:
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        car_crop = frame[y1:y2, x1:x2]

                        # kolor
                        detected_color = color_detector.classify_color(car_crop)
                        current_car_color = detected_color

                        # # rysowanie ramki i koloru na obrazie
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # cv2.putText(frame, f"Color: {detected_color}", (x1, y2 + 20),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        break

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()



@app.route('/')
def index():
    return render_template('index.html')

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
            # TODO: kolor??
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
    video_file = request.args.get('video', 'madzia.mp4')  # domyślnie amcia.mp4
    video_path = os.path.join('videos', video_file)
    if not os.path.exists(video_path):
        return "Video not found", 404
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_detection')
def latest_detection():
    return jsonify(current_plate_result)

# TODO: kolor auta
# @app.route('/latest_car_color')
# def latest_car_color():
#     return jsonify({'car_color': current_car_color})


if __name__ == '__main__':
    if not os.path.exists("static/logs"):
        os.makedirs("static/logs")
    app.run(debug=True, port=5004)
