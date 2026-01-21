# ============================================================
# mailer.py
# ============================================================
# PURPOSE:
# - Send CSV report via Gmail SMTP
# - Uses Streamlit secrets ONLY
#
# RULES:
# - No hardcoded credentials
# - No environment guessing
# - Explicit failures only
# - Streamlit-safe (Cloud + local)
# ============================================================

from pathlib import Path
import smtplib
from email.message import EmailMessage

import streamlit as st


def send_csv_email(csv_path: str) -> None:
    """
    Send CSV report via Gmail using Streamlit secrets.

    REQUIRED st.secrets:
    - GMAIL_USER
    - GMAIL_APP_PASSWORD
    - REPORT_RECEIVER_EMAIL
    """

    # --------------------------------------------------------
    # VALIDATE SECRETS
    # --------------------------------------------------------
    required_secrets = [
        "GMAIL_USER",
        "GMAIL_APP_PASSWORD",
        "REPORT_RECEIVER_EMAIL",
    ]

    for key in required_secrets:
        if key not in st.secrets:
            raise RuntimeError(f"Missing Streamlit secret: {key}")

    sender_email = st.secrets["GMAIL_USER"]
    app_password = st.secrets["GMAIL_APP_PASSWORD"]
    receiver_email = st.secrets["REPORT_RECEIVER_EMAIL"]

    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # --------------------------------------------------------
    # VALIDATE CSV PATH
    # --------------------------------------------------------
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not csv_path.is_file():
        raise ValueError(f"Path is not a file: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path.name}")

    # --------------------------------------------------------
    # BUILD EMAIL
    # --------------------------------------------------------
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "GIS Classification – Run Report"

    msg.set_content(
        "Attached is the CSV report for the latest GIS classification run.",
        subtype="plain",
        charset="utf-8",
    )

    with csv_path.open("rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="text",
            subtype="csv",
            filename=csv_path.name,
        )

    # --------------------------------------------------------
    # SEND EMAIL (EXPLICIT SMTP)
    # --------------------------------------------------------
    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        server.ehlo()

        if not server.has_extn("STARTTLS"):
            raise RuntimeError("SMTP server does not support STARTTLS")

        server.starttls()
        server.ehlo()

        server.login(sender_email, app_password)
        server.send_message(msg)
