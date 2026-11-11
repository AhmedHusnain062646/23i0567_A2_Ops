"""Task 5 - database trail. SQLite locally, PostgreSQL in Compose (same code, different DATABASE_URL)."""
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    filename = db.Column(db.String(255), nullable=False)       # sanitised name only; image bytes are NOT stored
    label = db.Column(db.String(20), nullable=False)           # "AI-generated" | "Real"
    confidence = db.Column(db.Float, nullable=False)           # confidence in the label shown
    prob_ai = db.Column(db.Float, nullable=False)              # raw sigmoid output
    threshold = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(120), nullable=False)
    model_version = db.Column(db.String(40), nullable=False)
    source = db.Column(db.String(10), nullable=False)          # local | remote (which inference path)
    latency_ms = db.Column(db.Integer)
