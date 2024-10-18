from http import client
import socket

from constants import DRIVE_LOOP_HZ


class HealthCheck:

    clientSocket = None

    def __init__(self, host_ip, host_port) -> None:
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSocket.connect((host_ip, host_port))

    def run(self):
        try:
            self.clientSocket.sendall("AMP_ALIVE".encode("ascii"))
            self.clientSocket.settimeout(1 / DRIVE_LOOP_HZ)
            res = self.clientSocket.recv(1024).decode()
            if res == "AMP_ALIVE":
                print("recieved ok from host")
                return True
            else:
                print("recieved not ok from host")
                return False
        except socket.timeout:
            print("socket timeout after 0.2 sec")
            return False
        except Exception as e:
            print(e)
            return False
