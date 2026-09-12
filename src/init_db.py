from src.database import Base, engine
from src.models_db import PredictionLog

Base.metadata.create_all(bind=engine)

print("Database tables created successfully.")
