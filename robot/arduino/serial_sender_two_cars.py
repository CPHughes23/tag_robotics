import serial
import time
import os
from pynput import keyboard
from dotenv import load_dotenv

load_dotenv()

SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
SERIAL_BAUD = int(os.getenv("SERIAL_BAUD", 9600))

ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD)
time.sleep(2)

# Tracked pressed keys
pressed_keys = set()

def send_car1():
    fwd = 'w' in pressed_keys
    rev = 's' in pressed_keys
    lft = 'a' in pressed_keys
    rgt = 'd' in pressed_keys

    if fwd:    ser.write(b'0')
    elif rev:  ser.write(b'1')
    else:      ser.write(b'f')

    if lft:    ser.write(b'2')
    elif rgt:  ser.write(b'3')
    else:      ser.write(b'r')

def send_car2():
    fwd = keyboard.Key.up    in pressed_keys
    rev = keyboard.Key.down  in pressed_keys
    lft = keyboard.Key.left  in pressed_keys
    rgt = keyboard.Key.right in pressed_keys

    if fwd:    ser.write(b'a')
    elif rev:  ser.write(b'b')
    else:      ser.write(b'g')

    if lft:    ser.write(b'c')
    elif rgt:  ser.write(b'd')
    else:      ser.write(b'h')

def on_press(key):
    # Quit
    try:
        if key.char == 'q':
            ser.write(b'4')  # stop car 1
            ser.write(b'8')  # stop car 2
            ser.close()
            return False
    except AttributeError:
        pass

    # Car 1: WASD
    try:
        k = key.char
        if k in ('w', 'a', 's', 'd') and k not in pressed_keys:
            pressed_keys.add(k)
            send_car1()
    except AttributeError:
        pass

    # Car 2: arrow keys
    if key in (keyboard.Key.up, keyboard.Key.down,
               keyboard.Key.left, keyboard.Key.right):
        if key not in pressed_keys:
            pressed_keys.add(key)
            send_car2()

def on_release(key):
    # Car 1 WASD
    try:
        k = key.char
        if k in ('w', 'a', 's', 'd') and k in pressed_keys:
            pressed_keys.discard(k)
            send_car1()
    except AttributeError:
        pass

    # Car 2 arrow keys
    if key in (keyboard.Key.up, keyboard.Key.down,
               keyboard.Key.left, keyboard.Key.right):
        if key in pressed_keys:
            pressed_keys.discard(key)
            send_car2()

print("Car 1: W/A/S/D to drive")
print("Car 2: Arrow keys to drive")
print("Q to quit\n")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()