from flask import Flask, render_template, Response, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from detection.yolo_detection import YOLODetection
from detection.ocr_detection import read_licence_plate
import datetime
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

# Kamera (ustaw na odpowiedni index kamery lub stream URL)
cap = cv2.VideoCapture(2)
detector = YOLODetection('../my_model/my_model.pt')  # <- podaj prawidłową ścieżkę


# Dummy detection (do podmiany na YOLO + OCR)
def dummy_detect_plate(frame):
    return "XYZ1234", frame


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Wykryj tablicę rejestracyjną
        plate_img = detector._process_frame(frame)  # możesz też użyć detect_plate(frame) jeśli wolisz

        if plate_img is not None:
            plate_number = read_licence_plate(plate_img)
        else:
            plate_number = "NO_PLATE"

        # Logika: czy tablica w bazie?
        session = Session()
        found = session.query(Plate).filter_by(plate_number=plate_number).first()
        status = 'GRANTED' if found else 'DENIED'

        session.add(Log(plate_number=plate_number, status=status))
        session.commit()
        session.close()

        # Nakładka na obraz
        msg = f"{plate_number} - {status}"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if status == 'GRANTED' else (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


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

if __name__ == '__main__':
    if not os.path.exists("static/logs"):
        os.makedirs("static/logs")
    app.run(debug=True)
