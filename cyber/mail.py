import smtplib
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# Get credentials from the config file
email_username = config.get('email', 'username')
email_password = config.get('email', 'password')
# Use SMTP_SSL and port 465
server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

# Start the server
server.ehlo()

with open('password.txt', 'r') as f:
    password = f.read()

server.login(email_username, email_password)

msg = MIMEMultipart()
msg['From'] = 'MEIR'
msg['To'] = 'ranconina@gmail.com'
msg['Subject'] = 'Just a test'

with open('message.txt', 'r') as f:
    message = f.read()

msg.attach(MIMEText(message, 'plain'))

filename = 'OIP.jpg'
attachment = open('OIP.jpg', 'rb')
p = MIMEBase('application', 'octet-stream')
p.set_payload(attachment.read())
encoders.encode_base64(p)
p.add_header('Content-Disposition', f'attachment; filename={filename}')
msg.attach(p)

text = msg.as_string()

server.sendmail('meiranconina@gmail.com', 'ranconina@gmail.com', text)

# Quit the server
server.quit()
