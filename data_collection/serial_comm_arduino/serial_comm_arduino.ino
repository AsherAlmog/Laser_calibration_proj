#include <ArduinoJson.h>
#include <SoftwareSerial.h>

// we should connect z stepper to X PIN, x stepper to E1 and theta to E2

// Define the stepper motor connections
#define X_STEP_PIN         54
#define X_DIR_PIN          55
#define X_ENABLE_PIN       38
#define Y_STEP_PIN         60
#define Y_DIR_PIN          61
#define Y_ENABLE_PIN       56
#define Z_STEP_PIN         46
#define Z_DIR_PIN          48
#define Z_ENABLE_PIN       62
#define E1_STEP_PIN        26
#define E1_DIR_PIN         28
#define E1_ENABLE_PIN      24
#define E2_STEP_PIN        36
#define E2_DIR_PIN         34
#define E2_ENABLE_PIN      30

#define MICRO_STEPS      32

// Define the number of steps per revolution for the stepper motors
#define STEPS_PER_REV     12800

// Define the number of rotations to perform
#define NUM_ROTATIONS      20

// Define the speed at which the motors should rotate (in steps per second)
#define SPEED              3*400*32

int32_t speedStep = 2*10^6/STEPS_PER_REV/2; // Speed/sec = second * num microseconds / num steps per revolution / delay twice
//SoftwareSerial mySerial(2, 3); // RX, TX


String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

void setup() {
  Serial.begin(9600);
  while (!Serial) continue; // Wait for serial connection
  // reserve 200 bytes for the inputString:
  inputString.reserve(200);
  // Set the enable pins for each motor                                                                                                                                                                                         
  pinMode(X_ENABLE_PIN, OUTPUT);
  digitalWrite(X_ENABLE_PIN, LOW);
  pinMode(Y_ENABLE_PIN, OUTPUT);
  digitalWrite(Y_ENABLE_PIN, LOW);
  pinMode(Z_ENABLE_PIN, OUTPUT);
  digitalWrite(Z_ENABLE_PIN, LOW);
  pinMode(E1_ENABLE_PIN, OUTPUT);
  digitalWrite(E1_ENABLE_PIN, LOW);
  pinMode(E2_ENABLE_PIN, OUTPUT);
  digitalWrite(E2_ENABLE_PIN, LOW);

  pinMode(Z_STEP_PIN, OUTPUT);
  digitalWrite(Z_STEP_PIN, LOW);

  pinMode(Z_DIR_PIN, OUTPUT);
  digitalWrite(Z_DIR_PIN, LOW);

  pinMode(E1_STEP_PIN, OUTPUT);
  digitalWrite(E1_STEP_PIN, LOW);

  pinMode(E1_DIR_PIN, OUTPUT);
  digitalWrite(E1_DIR_PIN, LOW);

    pinMode(E2_STEP_PIN, OUTPUT);
  digitalWrite(E2_STEP_PIN, LOW);

  pinMode(E2_DIR_PIN, OUTPUT);
  digitalWrite(E2_DIR_PIN, LOW);

}
int flag = 1;
int x = 0;
int theta = 0;
int z = 0;
int dirz = 0;
int dirx = 0;
int dirtz = 0;
void loop() {
  digitalWrite(X_DIR_PIN, LOW);
  digitalWrite(E1_DIR_PIN, LOW);
  digitalWrite(E2_DIR_PIN, LOW);
// *** if there's an isuue try initializing the pins to LOW here *** 
  if (Serial&&flag){
      Serial.println("ready"); 
      flag = 0;   
  }
  // Parse incoming string
  if (stringComplete) {
    StaticJsonDocument<250> incoming_doc;
    DeserializationError error = deserializeJson(incoming_doc, inputString);
    if (!error) {
      // Process incoming coordinates json file
      if (incoming_doc.containsKey("z")&& incoming_doc.containsKey("x")&& incoming_doc.containsKey("theta")&& incoming_doc.containsKey("dirz")&& incoming_doc.containsKey("dirx")&&incoming_doc.containsKey("dirtz")) {
        z = incoming_doc["z"].as<int>();
        theta = incoming_doc["theta"].as<int>();
        x = incoming_doc["x"].as<int>();
        dirz = incoming_doc["dirz"].as<int>();
        dirtz = incoming_doc["dirtz"].as<int>();
        dirx = incoming_doc["dirx"].as<int>();
        // choose which direction to spin based on the json input
        if(dirx==0){
          digitalWrite(E1_DIR_PIN, LOW);        
        }
        else{
          digitalWrite(E1_DIR_PIN, HIGH);
        }
        if(dirtz==0){
          digitalWrite(E2_DIR_PIN, LOW);          
        }
        else{
          digitalWrite(E2_DIR_PIN, HIGH);
        }
        if(dirz==0){
          digitalWrite(X_DIR_PIN, LOW);         
        }
        else{
          digitalWrite(X_DIR_PIN, HIGH);
        }
        //make rotations based on the json file input
        for (int k=0; k<z; k++){
          for (int i = 0; i<STEPS_PER_REV; i++){
            digitalWrite(X_STEP_PIN, LOW);
            delayMicroseconds(speedStep);
            digitalWrite(X_STEP_PIN, HIGH);
            delayMicroseconds(speedStep);
          }
          delay(50); // wait 0.1 sec between each spin
        }
        for (int k=0; k<x; k++){
          for (int i = 0; i<STEPS_PER_REV; i++){
            digitalWrite(E1_STEP_PIN, LOW);
            delayMicroseconds(speedStep);
            digitalWrite(E1_STEP_PIN, HIGH);
            delayMicroseconds(speedStep);
          }
          delay(50); // wait 0.1 sec between each spin
        }
        for (int k=0; k<theta; k++){
          for (int i = 0; i<STEPS_PER_REV; i++){
            digitalWrite(E2_STEP_PIN, LOW);
            delayMicroseconds(speedStep);
            digitalWrite(E2_STEP_PIN, HIGH);
            delayMicroseconds(speedStep);
          }
          delay(50); // wait 0.1 sec between each spin
        }
        delay(100); // wait 1 sec after finishing each movement
        // now tell the RPI that rotating is over
        Serial.println("moved");
      }
    }
    inputString = ""; // empty the input string to be able to recieve a new one
    stringComplete = false;
  }
  
}

void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}