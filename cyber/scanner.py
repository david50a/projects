#SYN scanning
#UDP Scan
#Comprehensive
import nmap
scanner=nmap.PortScanner()
print('Welcome to our NMAP Port Scanner ')
print('<---------------------------------------------->')
ip=input('Please the IP address that you want to scan: ')
print("The IP address is: ",ip)
resp=int(input("""\n Please enter the type of scan you want  to perform
        1. SYN ACK Scan
        2. UDP Scan
        3. Comprehesive Scan\n
"""))
print("You have selected the option: ",resp)
match resp:
    case 1:
        print('Nmap Version: ',scanner.nmap_version())
        scanner.scan(ip,'1-1024','-v -sS')
        print(scanner.scaninfo())
        print("IP staus: ",scanner[ip].state())
        print('IP protocal',scanner[ip].all_protocols())
        print("Open posts: ",scanner[ip]['tcp'].keys())
    case 2:
        print('Nmap Version: ',scanner.nmap_version())
        scanner.scan(ip,'1-1024','-v -sU')
        print(scanner.scaninfo())
        print("IP staus: ",scanner[ip].state())
        print('IP protocal',scanner[ip].all_protocols())
        print("Open posts: ",scanner[ip]['udp'].keys())
    case 3:
        print('Nmap Version: ',scanner.nmap_version())
        scanner.scan(ip,'1-1024','-v -sU -sV -sC -A -O')
        print(scanner.scaninfo())
        print("IP staus: ",scanner[ip].state())
        print('IP protocal',scanner[ip].all_protocols())
        print("Open posts: ",scanner[ip]['tcp'].keys())
    case _:
        print('Invalid value')
        
