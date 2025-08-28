from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv
import os

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

print("✅ DEBUG: EMAIL_ADDRESS:", EMAIL_ADDRESS)
print("✅ DEBUG: EMAIL_PASSWORD:", EMAIL_PASSWORD)

msg = EmailMessage()
msg['Subject'] = '✅ TEST: Python Email Working!'
msg['From'] = EMAIL_ADDRESS
msg['To'] = EMAIL_ADDRESS
msg.set_content('🎉 Your Python email works perfectly!')

try:
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
    print("✅ Test email sent successfully!")
except Exception as e:
    print("❌ ERROR:", e)
