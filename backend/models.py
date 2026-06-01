from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime
from database import Base

class Criterio3(Base):
    __tablename__ = "criterio_3"

    id              = Column(Integer, primary_key=True, index=True)
    imagen_path     = Column(String, nullable=False)   # local o URL futura
    prob_sop        = Column(Float, nullable=False)
    resultado       = Column(String, nullable=False)   # "Cumple" / "No cumple"
    validado        = Column(Boolean, default=False)   # médico confirma
    etiqueta_real   = Column(String, nullable=True)    # "Normal" / "SOP" — para reentrenar
    fecha           = Column(DateTime, default=datetime.utcnow)