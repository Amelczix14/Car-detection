from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import uvicorn

app = FastAPI()

SQLALCHEMY_DATABASE_URL = "postgresql://user:password@db:5432/parking_db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Vehicle(Base):
    __tablename__ = 'vehicles'
    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, unique=True, index=True)

class Entry(Base):
    __tablename__ = 'entries'
    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String)

Base.metadata.create_all(bind=engine)

class PlateNumber(BaseModel):
    plate_number: str

@app.post("/entry/")
def entry(plate: PlateNumber):
    session = SessionLocal()
    vehicle = session.query(Vehicle).filter(Vehicle.plate_number == plate.plate_number).first()

    if not vehicle:
        raise HTTPException(status_code=403, detail="Access Denied")

    new_entry = Entry(plate_number=plate.plate_number, status="in")
    session.add(new_entry)
    session.commit()
    session.close()
    return {"message": "Entry recorded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
