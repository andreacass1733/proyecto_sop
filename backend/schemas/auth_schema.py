from pydantic import BaseModel
from typing import Optional
from schemas.medico_schema import MedicoResponse

class LoginRequest(BaseModel):
    email: str
    password: str

class Verify2FARequest(BaseModel):
    session_id: str
    code: str

class Resend2FARequest(BaseModel):
    session_id: str

class InitialLoginResponse(BaseModel):
    requires_2fa: bool = True
    session_id: str
    email_masked: str
    telefono_masked: str
    codigo_demo: Optional[str] = None
    mensaje: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    email: str
    medico: MedicoResponse

class LogoutResponse(BaseModel):
    mensaje: str

class UserSessionResponse(BaseModel):
    user_id: str
    email: str
    medico: MedicoResponse

# ── Recuperación y Cambio de Contraseña (Seguridad en 2 Pasos) ─────────────
class ForgotPasswordRequest(BaseModel):
    email: str

class ForgotPasswordResponse(BaseModel):
    session_id: str
    email_masked: str
    mensaje: str

class ResetPasswordRequest(BaseModel):
    email: str
    code: str  # Código de 6 dígitos verificado por correo (obligatorio)
    new_password: str
    session_id: Optional[str] = None

class ResetPasswordResponse(BaseModel):
    mensaje: str
