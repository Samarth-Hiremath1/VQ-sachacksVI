from sqlalchemy import Column, DateTime, Integer, ForeignKey, DECIMAL
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from ..core.database import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    recording_id = Column(UUID(as_uuid=True), ForeignKey("recordings.id"), nullable=False, index=True)
    body_language_score = Column(DECIMAL(5, 2), nullable=True)
    speech_quality_score = Column(DECIMAL(5, 2), nullable=True)
    overall_score = Column(DECIMAL(5, 2), nullable=True)
    filler_word_count = Column(Integer, nullable=True)
    speaking_pace_wpm = Column(DECIMAL(6, 2), nullable=True)
    posture_score = Column(DECIMAL(5, 2), nullable=True)
    gesture_score = Column(DECIMAL(5, 2), nullable=True)
    eye_contact_score = Column(DECIMAL(5, 2), nullable=True)
    recommendations = Column(JSONB, nullable=True)
    detailed_metrics = Column(JSONB, nullable=True)
    processed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    recording = relationship("Recording", back_populates="analysis_results")

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, recording_id={self.recording_id}, overall_score={self.overall_score})>"