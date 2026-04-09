import serial
import time
from pynput import keyboard

ser = serial.Serial('/dev/cu.usbserial-2110', 9600)
time.sleep(2)

STOP = b'4'

KEY_COMMANDS = {
    'w': b'0',  # forward
    'a': b'2',  # left
    's': b'1',  # reverse
    'd': b'3',  # right
}

pressed_keys = set()

def send_command():
    fwd = 'w' in pressed_keys
    rev = 's' in pressed_keys
    lft = 'a' in pressed_keys
    rgt = 'd' in pressed_keys

    # forward/back axis
    if fwd:        ser.write(b'0')
    elif rev:      ser.write(b'1')
    else:          ser.write(b'f')  # stop forward/back axis only

    # left/right axis
    if lft:        ser.write(b'2')
    elif rgt:      ser.write(b'3')
    else:          ser.write(b'r')  # stop left/right axis only

def on_press(key):
    try:
        k = key.char
        if k in KEY_COMMANDS:
            if k not in pressed_keys:
                pressed_keys.add(k)
                send_command()
        elif k == 'q':
            ser.write(STOP)
            ser.close()
            return False  # stop listener
    except AttributeError:
        pass  # ignore special keys

def on_release(key):
    try:
        k = key.char
        if k in KEY_COMMANDS and k in pressed_keys:
            pressed_keys.discard(k)
            send_command()
    except AttributeError:
        pass

print("Controls: W/A/S/D to drive, Q to quit")
print("Hold a key to move, release to stop.\n")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()