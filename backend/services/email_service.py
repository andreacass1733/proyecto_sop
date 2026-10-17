"""
Servicio de Notificaciones y Envío de Correo Electrónico — Doble Factor (2FA) y Recuperación.
═══════════════════════════════════════════════════════════════════════════════
Permite despachar correos electrónicos vía SMTP (Gmail, Outlook, Brevo, etc.)
directamente a la bandeja de entrada del médico especialista.
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import make_msgid, formatdate

logger = logging.getLogger("email_service")

def _obtener_config_smtp():
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = (os.getenv("SMTP_USER", "") or "").strip()
    raw_pwd = (os.getenv("SMTP_PASSWORD", "") or "").strip()
    # Si es contraseña de aplicaciones de Google (16 letras separadas por espacios), unificarla
    password = raw_pwd.replace(" ", "") if len(raw_pwd.replace(" ", "")) == 16 else raw_pwd
    remitente = os.getenv("SMTP_FROM", user or "seguridad@sop-rotterdam.com")
    return host, port, user, password, remitente


def enviar_email_codigo_2fa(destinatario: str, codigo: str, nombre_medico: str = "Dr. Rodrigo Espinoza") -> bool:
    """
    Envía un correo electrónico con el código de verificación de 6 dígitos (2FA)
    con diseño clínico y cumplimiento de normativas de seguridad médica HIPAA.
    """
    host, port, user, password, remitente = _obtener_config_smtp()

    if not user or not password:
        logger.warning(
            f"[EMAIL SERVICE] Faltan credenciales SMTP_USER / SMTP_PASSWORD en backend/.env. "
            f"El código 2FA emitido es {codigo} para '{destinatario}'. Configure Gmail App Password en .env para entrega automática."
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Código de Seguridad 2FA: {codigo} — Sistema SOP Rotterdam"
        msg["From"] = f"Sistema SOP Rotterdam <{remitente}>"
        msg["To"] = destinatario
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="sop-rotterdam.com")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 20px; border: 1px solid #e2e8f0; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }}
            .header {{ text-align: center; margin-bottom: 24px; }}
            .badge {{ display: inline-block; background: #ffe4e6; color: #e11d48; font-size: 11px; font-weight: 800; padding: 4px 12px; border-radius: 9999px; text-transform: uppercase; letter-spacing: 1px; }}
            .title {{ font-size: 20px; font-weight: 900; color: #0f172a; margin-top: 12px; }}
            .code-box {{ background: #f1f5f9; border: 2px dashed #cbd5e1; border-radius: 16px; padding: 20px; text-align: center; margin: 24px 0; }}
            .code {{ font-size: 32px; font-weight: 900; letter-spacing: 8px; color: #7c3aed; font-family: monospace; }}
            .footer {{ font-size: 11px; color: #94a3b8; text-align: center; border-top: 1px solid #f1f5f9; padding-top: 16px; margin-top: 24px; }}
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <span class="badge">Seguridad Clínica HIPAA / 2FA</span>
              <h1 class="title">Verificación de Doble Factor</h1>
              <p style="font-size: 13px; color: #64748b; margin-top: 4px;">Estimado/a {nombre_medico},</p>
            </div>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
              Ha solicitado acceder al <strong>Sistema de Detección y Clasificación de SOP Rotterdam</strong>. Para proteger la confidencialidad de los expedientes médicos, introduzca el siguiente código de verificación:
            </p>
            <div class="code-box">
              <div class="code">{codigo}</div>
              <p style="font-size: 11px; color: #64748b; margin-top: 8px; font-weight: 600;">Válido durante 5 minutos</p>
            </div>
            <p style="font-size: 12px; color: #64748b; line-height: 1.5;">
              Si usted no solicitó este inicio de sesión, ignore este mensaje de inmediato y notifique al área de seguridad clínica.
            </p>
            <div class="footer">
              Sistema de Evaluación Diagnóstica SOP • Cumplimiento Normativo HIPAA / GDPR
            </div>
          </div>
        </body>
        </html>
        """

        part = MIMEText(html_content, "html")
        msg.attach(part)

        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=12)
        else:
            server = smtplib.SMTP(host, port, timeout=12)
            server.starttls()

        server.login(user, password)
        server.sendmail(remitente, [destinatario], msg.as_string())
        server.quit()

        logger.info(f"[EMAIL SUCCESS] Código 2FA enviado satisfactoriamente vía SMTP a '{destinatario}'.")
        return True

    except Exception as e:
        logger.error(f"[EMAIL ERROR] Falló el envío de correo SMTP a '{destinatario}': {e}")
        return False


def enviar_email_recuperacion(destinatario: str, codigo: str, nombre_medico: str = "Dr. Rodrigo Espinoza") -> bool:
    """
    Envía un correo electrónico con el código de recuperación de contraseña.
    """
    host, port, user, password, remitente = _obtener_config_smtp()

    if not user or not password:
        logger.warning(
            f"[EMAIL SERVICE] Faltan credenciales SMTP en .env. Código de recuperación: {codigo} para '{destinatario}'."
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Recuperación de Contraseña: {codigo} — Sistema SOP Rotterdam"
        msg["From"] = f"Sistema SOP Rotterdam <{remitente}>"
        msg["To"] = destinatario
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="sop-rotterdam.com")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 20px; border: 1px solid #e2e8f0; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }}
            .header {{ text-align: center; margin-bottom: 24px; }}
            .badge {{ display: inline-block; background: #fef3c7; color: #b45309; font-size: 11px; font-weight: 800; padding: 4px 12px; border-radius: 9999px; text-transform: uppercase; letter-spacing: 1px; }}
            .title {{ font-size: 20px; font-weight: 900; color: #0f172a; margin-top: 12px; }}
            .code-box {{ background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 16px; padding: 20px; text-align: center; margin: 24px 0; }}
            .code {{ font-size: 32px; font-weight: 900; letter-spacing: 8px; color: #e11d48; font-family: monospace; }}
            .footer {{ font-size: 11px; color: #94a3b8; text-align: center; border-top: 1px solid #f1f5f9; padding-top: 16px; margin-top: 24px; }}
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <span class="badge">Seguridad de Acceso Clínico</span>
              <h1 class="title">Restablecimiento de Contraseña</h1>
              <p style="font-size: 13px; color: #64748b; margin-top: 4px;">Estimado/a {nombre_medico},</p>
            </div>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
              Ha solicitado restablecer su contraseña de acceso al <strong>Sistema SOP Rotterdam</strong>. Ingrese el siguiente código en el portal para crear su nueva contraseña:
            </p>
            <div class="code-box">
              <div class="code">{codigo}</div>
              <p style="font-size: 11px; color: #64748b; margin-top: 8px; font-weight: 600;">Válido durante 10 minutos</p>
            </div>
            <p style="font-size: 12px; color: #64748b; line-height: 1.5;">
              Si usted no realizó esta solicitud, contacte de inmediato con la administración del sistema.
            </p>
            <div class="footer">
              Sistema de Evaluación Diagnóstica SOP • Cumplimiento Normativo HIPAA / GDPR
            </div>
          </div>
        </body>
        </html>
        """

        part = MIMEText(html_content, "html")
        msg.attach(part)

        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=12)
        else:
            server = smtplib.SMTP(host, port, timeout=12)
            server.starttls()

        server.login(user, password)
        server.sendmail(remitente, [destinatario], msg.as_string())
        server.quit()

        logger.info(f"[EMAIL SUCCESS] Código de recuperación enviado satisfactoriamente a '{destinatario}'.")
        return True

    except Exception as e:
        logger.error(f"[EMAIL ERROR] Falló el envío de correo de recuperación a '{destinatario}': {e}")
        return False
