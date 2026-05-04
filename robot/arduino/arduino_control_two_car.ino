void setup() {
  // Car 1: pins 2-5
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);

  // Car 2: pins 7-10
  pinMode(7, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);

  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    // Car 1 (pins 2-5)
    if (cmd == '4') {
      // stop car 1
      digitalWrite(2, LOW);
      digitalWrite(3, LOW);
      digitalWrite(4, LOW);
      digitalWrite(5, LOW);
      digitalWrite(LED_BUILTIN, LOW);
    }

    if (cmd == '0') { digitalWrite(2, HIGH); digitalWrite(5, LOW); }  // forward on
    if (cmd == '1') { digitalWrite(5, HIGH); digitalWrite(2, LOW); }  // reverse on
    if (cmd == 'f') { digitalWrite(2, LOW);  digitalWrite(5, LOW); }  // forward/back off

    if (cmd == '2') { digitalWrite(4, HIGH); digitalWrite(3, LOW); }  // left on
    if (cmd == '3') { digitalWrite(3, HIGH); digitalWrite(4, LOW); }  // right on
    if (cmd == 'r') { digitalWrite(4, LOW);  digitalWrite(3, LOW); }  // left/right off

    // Car 2 (pins 7-10)
    if (cmd == '8') {
      // stop car 2
      digitalWrite(7,  LOW);
      digitalWrite(8,  LOW);
      digitalWrite(9,  LOW);
      digitalWrite(10, LOW);
      digitalWrite(LED_BUILTIN, LOW);
    }

    if (cmd == 'a') { digitalWrite(7,  HIGH); digitalWrite(10, LOW); }  // forward on
    if (cmd == 'b') { digitalWrite(10, HIGH); digitalWrite(7,  LOW); }  // reverse on
    if (cmd == 'g') { digitalWrite(7,  LOW);  digitalWrite(10, LOW); }  // forward/back off

    if (cmd == 'c') { digitalWrite(9,  HIGH); digitalWrite(8,  LOW); }  // left on
    if (cmd == 'd') { digitalWrite(8,  HIGH); digitalWrite(9,  LOW); }  // right on
    if (cmd == 'h') { digitalWrite(9,  LOW);  digitalWrite(8,  LOW); }  // left/right off
  }
}
