#include <Arduino.h>


#define electromagnetPort 2
char activateButton = 'p';
char releaseButton = 'r';
char serialInput;


void setup() {
    // put your setup code here, to run once:
    Serial.begin(9600);
    pinMode(electromagnetPort, OUTPUT);
}

void loop() {
    // check for drop button press
    if (Serial.available() > 0) {
        serialInput = Serial.read();
        Serial.println(serialInput);
        if (serialInput == activateButton) {
            digitalWrite(electromagnetPort, HIGH); // turn on electromagnet
        }
        else if (serialInput == releaseButton) {
            digitalWrite(electromagnetPort, LOW); // turn off electromagnet
        }
        serialInput = ' ';
    }
}
