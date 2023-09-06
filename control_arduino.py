import serial
from pyinput import keyboard

global ser

def on_press(key):
	try:
		if key.char == 'p':
			ser.write(b'p')
			print("sent p")
            
        #if i press "r" send "r" to arduino
		if key.char == 'r':
			ser.write(b'r')
			print("sent r")
        
        #if q is pressed, quit
		if key.char == 'q':
			print("quit")
			ser.close()
			break
	except AttributeError:
		pass

def main():
    ser = serial.Serial('COM6', 9600)
    #if i press "p" send "p" to arduino
    

	with keyboard.Listener(on_press =on_press, on_release=on_release) as listener:
		listener.join()
        


if __name__ == '__main__':
    main()
   
