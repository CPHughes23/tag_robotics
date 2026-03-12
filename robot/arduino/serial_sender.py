import serial
import time

# This connects to device 110 over serial on 9600 (find this info in arduino IDE)
ser = serial.Serial('/dev/cu.usbserial-110', 9600)
time.sleep(2) # wait for arduino to reset

def press():
    ser.write(b'1')

def release():
    ser.write(b'0')

def main():
    while True:
        cmd = input("Enter command (1 for on, 0 for off, q to quit)")

        if cmd == '1':
            press()
        elif cmd == '0':
            release()
        elif cmd == 'q':
            break
        else:
            print("Unknown command")

if __name__ == '__main__':
    main()