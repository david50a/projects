import scapy.all as scapy
def scanner(ip):
    request=scapy.ARP()
    request.pdst=ip
    print(request.summary())
    scapy.ls(scapy.ARP())
    broadcast=scapy.Ether()
    broadcast.dst="ff:ff:ff:ff:ff:ff"
    broadcast.show()
    request_broadcast=broadcast/request
    request_broadcast.show()
    resp1,resp2=scapy.srp(request_broadcast,timeout=2)
    [print("IP: ",i[1].psrc,"MAC: ",i[1].hwsrc) for i in resp1]
scanner('10.10.43.254/24')
    
        
