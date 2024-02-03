const int ledPin = 13;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW); // Ensure the LED is initially off
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();
    
    if (receivedChar == '1') {
      digitalWrite(ledPin, HIGH); 
      delay(2000);
      digitalWrite(ledPin, LOW);
    } else if (receivedChar == '0') {
      digitalWrite(ledPin, LOW);   // Turn off the LED
    }
  }
}
