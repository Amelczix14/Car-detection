from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import os
import cv2
from main_easyocr import read_licence_plate
from yolo_detect import YOLODetection

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👉 Połączenie z bazą danych
def connect_to_db():
    return psycopg2.connect(
        host="db",
        database="parking_db",
        user="user",
        password="password",
        port=5432
    )


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    temp_filename = "input.jpg"
    with open(temp_filename, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # 🟢 Wykorzystanie Twojego YOLO:
    yolo = YOLODetection()
    plate_image = yolo.detect_plate(temp_filename)
    if plate_image is None:
        os.remove(temp_filename)
        return {"error": "Nie wykryto tablicy."}

    plate_number = read_licence_plate(plate_image)
    os.remove(temp_filename)

    conn = connect_to_db()
    cursor = conn.cursor()

    # 👉 Dodanie do bazy lub sprawdzenie czy auto już jest
    cursor.execute("INSERT INTO vehicles (plate_number) VALUES (%s) ON CONFLICT (plate_number) DO NOTHING", (plate_number,))
    cursor.execute("SELECT id FROM vehicles WHERE plate_number = %s", (plate_number,))
    vehicle_id = cursor.fetchone()[0]

    cursor.execute("SELECT is_parked FROM parking_status WHERE vehicle_id = %s", (vehicle_id,))
    result = cursor.fetchone()

    if result and result[0]:
        action = 'OUT'
        cursor.execute("UPDATE parking_status SET is_parked = FALSE WHERE vehicle_id = %s", (vehicle_id,))
    else:
        action = 'IN'
        cursor.execute("INSERT INTO parking_status (vehicle_id, is_parked) VALUES (%s, TRUE) ON CONFLICT (vehicle_id) DO UPDATE SET is_parked = TRUE", (vehicle_id,))

    cursor.execute("INSERT INTO logs (vehicle_id, action) VALUES (%s, %s)", (vehicle_id, action))
    conn.commit()
    cursor.close()
    conn.close()

    return {"plate_number": plate_number, "action": action}


@app.get("/status/")
def parking_status():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.plate_number, l.timestamp
        FROM parking_status p
        JOIN vehicles v ON p.vehicle_id = v.id
        JOIN logs l ON l.vehicle_id = p.vehicle_id
        WHERE p.is_parked = TRUE
        ORDER BY l.timestamp DESC;
    """)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"plate_number": row[0], "timestamp": row[1]} for row in data]

# 👉 HISTORIA WJAZDÓW I WYJAZDÓW
@app.get("/logs/")
def logs():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT l.id, v.plate_number, l.timestamp, l.action
        FROM logs l
        JOIN vehicles v ON l.vehicle_id = v.id
        ORDER BY l.timestamp DESC;
    """)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"id": row[0], "plate_number": row[1], "timestamp": row[2], "action": row[3]} for row in data]
