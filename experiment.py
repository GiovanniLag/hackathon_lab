import serial
import msvcrt

def main():
    ser = serial.Serial('COM6', 9600)
    #if i press "p" send "p" to arduino
    while True:
        if ord(msvcrt.getch()) == 112:
            ser.write(b'p')
            print("sent p")
            
        #if i press "r" send "r" to arduino
        if ord(msvcrt.getch()) == 114:
            ser.write(b'r')
            print("sent r")
        
        #if q is pressed, quit
        if ord(msvcrt.getch()) == 113:
            print("quit")
            break

    ser.close()

if __name__ == '__main__':
    main()