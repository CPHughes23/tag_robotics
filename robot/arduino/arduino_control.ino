void setup() {
    pinMode(10, OUTPUT);
    pinMode(LED_BUILTIN, OUTPUT)
    Serial.begin(9600)
}

void loop() {
    if (Serial.available()) {
        char cmd = Serial.read();
        if (cmd == '1') {
            digitalWrite(10, HIGH);
            digitalWrite(LED_BUILTIN, HIGH);
        }
        if (cmd == '0') {
            digitalWrite(10, LOW);
            digitalWrite(LED_BUILTIN, LOW);
        }
    }
}