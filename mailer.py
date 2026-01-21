# ============================================================
# mailer.py
# ============================================================
# PURPOSE:
# - Send CSV report via email
# - Gmail / SMTP compatible
#
# DESIGN RULES:
# - NO retries
# - NO silent fallback
# - NO environment guessing
# - Explicit failure only
#
# Python: 3.8 / 3.9 compatible
# ============================================================

from pathlib import Path
import smtplib
from email.message import EmailMessage


def send_csv_email(
    csv_path,
    sender_email,
    receiver_email,
    smtp_server,
    smtp_port,
    app_password
) -> None:
    """
    Send CSV file as email attachment.

    Raises exception on failure.
    """

    csv_path = Path(csv_path)

    # --------------------------------------------------------
    # PATH VALIDATION
    # --------------------------------------------------------
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not csv_path.is_file():
        raise ValueError(f"CSV path is not a file: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"File is not a CSV: {csv_path.name}")

    # --------------------------------------------------------
    # EMAIL MESSAGE
    # --------------------------------------------------------
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "GIS Classification Run Report"
    msg.set_content(
        "Attached is the CSV report for the latest GIS classification session.",
        subtype="plain",
        charset="utf-8"
    )

    with open(csv_path, "rb") as f:
        data = f.read()

    msg.add_attachment(
        data,
        maintype="text",
        subtype="csv",
        filename=csv_path.name
    )

    # --------------------------------------------------------
    # SMTP SEND (EXPLICIT)
    # --------------------------------------------------------
    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        server.ehlo()
        if server.has_extn("STARTTLS"):
            server.starttls()
            server.ehlo()
        else:
            raise RuntimeError("SMTP server does not support STARTTLS")

        server.login(sender_email, app_password)
        server.send_message(msg)
