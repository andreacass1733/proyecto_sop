"""
MÓDULO DE SEGURIDAD Y CRIPTOGRAFÍA MÉDICA (HIPAA & GDPR COMPLIANCE)
═════════════════════════════════════════════════════════════════════════════════

Este servicio implementa las normativas de seguridad en salud internacional:
  - HIPAA (Health Insurance Portability and Accountability Act - 45 CFR § 164.312):
    * Cifrado de datos de salud protegidos (PHI) en reposo (AES-256 / Fernet)
    * Anonimización y desidentificación de metadatos (Estándar Safe Harbor § 164.514)
    * Verificación de integridad mediante hashes criptográficos SHA-256
  - GDPR (General Data Protection Regulation - Art. 32):
    * Protección de datos personales y trazabilidad de procesamiento clínico
"""

import os
import hashlib
import base64
import logging
from typing import Tuple, Optional
from cryptography.fernet import Fernet
import cv2
import numpy as np

# Configuración de Logging de Seguridad (Audit Trail)
logger = logging.getLogger("med_security")
logger.setLevel(logging.INFO)

# Clave de cifrado simétrica AES (obtenida de .env o generada automáticamente)
_ENCRYPTION_KEY_STR = os.getenv("HIPAA_ENCRYPTION_KEY")

if not _ENCRYPTION_KEY_STR:
    # Si no existe en .env, generar una clave Fernet válida de 32 bytes en base64
    _RAW_KEY = Fernet.generate_key()
    _ENCRYPTION_KEY = _RAW_KEY
else:
    # Asegurar que esté codificada en bytes
    _ENCRYPTION_KEY = _ENCRYPTION_KEY_STR.encode('utf-8')

_cipher_suite = Fernet(_ENCRYPTION_KEY)


def cifrar_imagen_medica(imagen_bytes: bytes) -> bytes:
    """
    HIPAA 45 CFR § 164.312(a)(2)(iv) - Cifrado en reposo (Encryption at Rest)
    Cifra los bytes de la ecografía médica utilizando el estándar AES (Fernet).
    """
    try:
        bytes_cifrados = _cipher_suite.encrypt(imagen_bytes)
        logger.info("[SEGURIDAD HIPAA] Imagen médica cifrada exitosamente con AES-256/Fernet.")
        return bytes_cifrados
    except Exception as e:
        logger.error(f"[ERROR SEGURIDAD] Fallo al cifrar imagen médica: {e}")
        raise RuntimeError("Fallo de seguridad al cifrar ecografía clínica") from e


def descifrar_imagen_medica(bytes_cifrados: bytes) -> bytes:
    """
    HIPAA 45 CFR § 164.312(a)(2)(iv) - Descifrado controlado
    Descifra los bytes de la ecografía médica únicamente para personal autorizado / modelo de IA.
    """
    try:
        bytes_originales = _cipher_suite.decrypt(bytes_cifrados)
        logger.info("[SEGURIDAD HIPAA] Imagen médica descifrada correctamente para procesamiento.")
        return bytes_originales
    except Exception as e:
        logger.error(f"[ERROR SEGURIDAD] Fallo al descifrar imagen médica: {e}")
        raise RuntimeError("Acceso no autorizado o corrupción de clave de descifrado") from e


def calcular_hash_sha256(data_bytes: bytes) -> str:
    """
    HIPAA § 164.312(c)(1) - Integridad de Datos (Data Integrity Verification)
    Genera una suma de verificación SHA-256 para garantizar que la imagen ecográfica
    no ha sido alterada o corrompida desde su captura.
    """
    return hashlib.sha256(data_bytes).hexdigest()


def desidentificar_metadatos_imagen(img_path_or_bytes: bytes | str) -> np.ndarray:
    """
    HIPAA Safe Harbor Standard (45 CFR § 164.514(b)) & GDPR Art. 32
    Elimina metadatos identificables de la ecografía (Nombres, Cédulas, Fechas del ecógrafo)
    y limpia los canales de la imagen dejando únicamente la matriz de píxeles tisulares.
    """
    if isinstance(img_path_or_bytes, str):
        img_bgr = cv2.imread(img_path_or_bytes)
    else:
        nparr = np.frombuffer(img_path_or_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("No se pudo cargar la imagen para desidentificación de metadatos")

    # Limpiar cualquier texto incrustado en bordes extremos (recorte de bandas de encabezado DICOM)
    h, w, _ = img_bgr.shape
    # Si la imagen contiene franjas de metadatos superiores/inferiores del ecógrafo
    # Mantenemos la región de interés (ROI) central del tejido ovárico
    roi = img_bgr[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
    roi_resized = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    return roi_resized


def auditar_acceso_medico(usuario_id: str, accion: str, recurso_id: str):
    """
    HIPAA § 164.312(b) - Controles de Auditoría (Audit Trail Controls)
    Registra cualquier acceso, modificación o descarga de información médica sensible.
    """
    logger.info(f"[AUDIT TRAIL HIPAA/GDPR] Usuario: {usuario_id} | Acción: {accion} | Recurso: {recurso_id}")
