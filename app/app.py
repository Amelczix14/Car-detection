from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from detection.yolo_detection import YOLODetection
from detection.ocr_detection import read_licence_plate
import datetime
import time
import cv2
import os

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

cap = cv2.VideoCapture(2)
detector = YOLODetection('../my_model/my_model.pt')

# Dummy detection (do podmiany na YOLO + OCR)
def dummy_detect_plate(frame):
    return "XYZ1234", frame

def generate_frames(video_path=None):
    cap = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(2)  # kamera domyślna

    last_granted_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        plate_img = detector._process_frame(frame)
        plate_number = "NO_PLATE"
        status = "PROCESSING"

        if plate_img is not None and (time.time() - last_granted_time > 10):
            reads = []
            for _ in range(3):
                plate_text, conf = read_licence_plate(plate_img)
                if plate_text:
                    reads.append(plate_text)
            if reads:
                plate_number = max(set(reads), key=reads.count)

            session = Session()
            found = session.query(Plate).filter_by(plate_number=plate_number).first()
            status = 'GRANTED' if found else 'DENIED'
            session.add(Log(plate_number=plate_number, status=status))
            session.commit()
            session.close()

            if status == "GRANTED":
                last_granted_time = time.time()

            current_plate_result["plate"] = plate_number
            current_plate_result["status"] = status
            current_plate_result["timestamp"] = time.time()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
    video_file = request.args.get('video', 'amcia.mp4')  # domyślnie amcia.mp4
    video_path = os.path.join('videos', video_file)
    if not os.path.exists(video_path):
        return "Video not found", 404
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

current_plate_result = {"plate": None, "status": None, "timestamp": None}

@app.route('/latest_detection')
def latest_detection():
    return jsonify(current_plate_result)


if __name__ == '__main__':
    if not os.path.exists("static/logs"):
        os.makedirs("static/logs")
    app.run(debug=True)
