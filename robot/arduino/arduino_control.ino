void setup() {
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(7, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    if (cmd == '4') {
      // stop all
      digitalWrite(4, LOW);
      digitalWrite(5, LOW);
      digitalWrite(6, LOW);
      digitalWrite(7, LOW);
      digitalWrite(LED_BUILTIN, LOW);
    }

    // forward/back pins
    if (cmd == '0') { digitalWrite(4, HIGH); digitalWrite(7, LOW); }  // forward on
    if (cmd == '1') { digitalWrite(7, HIGH); digitalWrite(4, LOW); }  // reverse on
    if (cmd == 'f') { digitalWrite(4, LOW);  digitalWrite(7, LOW); }  // forward/back off

    // left/right pins
    if (cmd == '2') { digitalWrite(6, HIGH); digitalWrite(5, LOW); }  // left on
    if (cmd == '3') { digitalWrite(5, HIGH); digitalWrite(6, LOW); }  // right on
    if (cmd == 'r') { digitalWrite(6, LOW);  digitalWrite(5, LOW); }  // left/right off
  }
}