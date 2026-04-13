/*
 * ============================================================
 *  Smart Face Identity Base — ESP32 Servo + PIR Controller
 * ============================================================
 *
 *  Hardware:
 *    - ESP32 DevKit (NodeMCU-32S or similar)
 *    - Servo motor (SG90 / MG996R) on GPIO 14
 *    - PIR sensor (HC-SR501)      on GPIO 13
 *
 *  Communication:
 *    Serial USB @ 115200 baud, line-based text protocol.
 *
 *  Commands FROM Python → ESP32:
 *    TRACK:<angle>   — Move servo to <angle> degrees (face tracking)
 *    SCAN            — Resume autonomous scanning sweep
 *    STOP            — Stop servo and go idle
 *    PIR_ON          — Enable PIR sensor readings
 *    PIR_OFF         — Disable PIR sensor readings
 *
 *  Messages FROM ESP32 → Python:
 *    MOTION:1        — PIR triggered (motion detected)
 *    MOTION:0        — No motion for timeout period
 *    ANGLE:<angle>   — Current servo angle (sent periodically)
 *    STATE:<state>   — Current state: IDLE / SCANNING / TRACKING
 *
 * ============================================================
 */

#include <ESP32Servo.h>

// ─── Pin Configuration ─────────────────────────────────────
#define SERVO_PIN       14
#define PIR_PIN         13

// ─── Servo Configuration ───────────────────────────────────
#define SERVO_MIN_ANGLE  30       // Leftmost sweep angle
#define SERVO_MAX_ANGLE  150      // Rightmost sweep angle
#define SERVO_CENTER     90       // Center / home position
#define SERVO_MIN_PULSE  500      // Minimum pulse width (µs)
#define SERVO_MAX_PULSE  2400     // Maximum pulse width (µs)

// ─── Timing Configuration (milliseconds) ───────────────────
#define SCAN_STEP_DELAY     40    // ms between 1° steps during scanning (smoother sweep)
#define TRACK_STEP_DELAY    15    // ms between steps during tracking
#define PIR_TIMEOUT         10000 // ms of no motion before going IDLE
#define ANGLE_REPORT_INTERVAL 200 // ms between ANGLE reports to Python
#define PIR_DEBOUNCE_MS     2000  // PIR debounce window (prevent false triggers)

// ─── Tracking Smoothness ───────────────────────────────────
// Maximum degrees the servo can move per tracking step.
// Larger values = faster but jerkier. Smaller = smoother.
#define TRACK_MAX_STEP      2.0   // max degrees per step
#define TRACK_MIN_STEP      0.3   // min degrees per step (prevents creeping)

// ─── State Machine ─────────────────────────────────────────
enum State {
  STATE_IDLE,
  STATE_SCANNING,
  STATE_TRACKING
};

State currentState = STATE_IDLE;

// ─── Global Variables ──────────────────────────────────────
Servo servo;

float currentAngle  = SERVO_CENTER;  // Actual servo position (float for sub-degree)
int   targetAngle   = SERVO_CENTER;  // Desired target angle
int   scanDirection = 1;             // 1 = increasing angle, -1 = decreasing

unsigned long lastStepTime       = 0;
unsigned long lastMotionTime     = 0;
unsigned long lastAngleReport    = 0;
unsigned long lastPIRTrigger     = 0;

bool motionActive = false;
bool prevPIRState = LOW;
bool pirEnabled   = true;           // Can be toggled by Python via PIR_ON / PIR_OFF

// ─── Serial Input Buffer ───────────────────────────────────
String serialBuffer = "";

// ─── Function Declarations ─────────────────────────────────
void handleSerialInput();
void updateStateMachine();
void reportAngle();
void setState(State newState);

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  // Configure PIR
  pinMode(PIR_PIN, INPUT);

  // Attach servo
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE);

  // Move to center on boot
  currentAngle = SERVO_CENTER;
  targetAngle  = SERVO_CENTER;
  servo.write(SERVO_CENTER);

  delay(500); // Let servo settle

  Serial.println("STATE:IDLE");
  Serial.print("ANGLE:");
  Serial.println(SERVO_CENTER);
}

// ============================================================
//  MAIN LOOP
// ============================================================
void loop() {
  // 1. Read serial commands from Python
  handleSerialInput();

  // 2. Read PIR sensor (only if enabled)
  if (pirEnabled) {
    bool pirReading = digitalRead(PIR_PIN);

    // Debounce PIR — only react to rising edges with a cooldown
    if (pirReading == HIGH && prevPIRState == LOW) {
      if (millis() - lastPIRTrigger > PIR_DEBOUNCE_MS) {
        lastPIRTrigger = millis();
        lastMotionTime = millis();

        if (!motionActive) {
          motionActive = true;
          Serial.println("MOTION:1");

          // If idle, start scanning
          if (currentState == STATE_IDLE) {
            setState(STATE_SCANNING);
          }
        }
      }
    }
    prevPIRState = pirReading;

    // Keep motion timer alive while PIR reads HIGH
    if (pirReading == HIGH) {
      lastMotionTime = millis();
    }

    // Check motion timeout
    if (motionActive && (millis() - lastMotionTime > PIR_TIMEOUT)) {
      motionActive = false;
      Serial.println("MOTION:0");

      // Only go idle if we're scanning (not tracking a face)
      if (currentState == STATE_SCANNING) {
        setState(STATE_IDLE);
      }
    }
  }

  // 3. Run the state machine
  updateStateMachine();

  // 4. Periodically report angle
  if (millis() - lastAngleReport > ANGLE_REPORT_INTERVAL) {
    reportAngle();
    lastAngleReport = millis();
  }
}

// ============================================================
//  SERIAL COMMAND PARSER
// ============================================================
void handleSerialInput() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      serialBuffer.trim();

      if (serialBuffer.length() > 0) {
        // Parse command
        if (serialBuffer.startsWith("TRACK:")) {
          int angle = serialBuffer.substring(6).toInt();
          angle = constrain(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
          targetAngle = angle;
          setState(STATE_TRACKING);
        }
        else if (serialBuffer == "SCAN") {
          setState(STATE_SCANNING);
        }
        else if (serialBuffer == "STOP") {
          setState(STATE_IDLE);
        }
        else if (serialBuffer == "PIR_ON") {
          pirEnabled = true;
          Serial.println("PIR:ON");
        }
        else if (serialBuffer == "PIR_OFF") {
          pirEnabled = false;
          motionActive = false;
          Serial.println("MOTION:0");
          Serial.println("PIR:OFF");
        }
      }

      serialBuffer = "";
    } else {
      serialBuffer += c;
    }
  }
}

// ============================================================
//  STATE MACHINE UPDATE
// ============================================================
void updateStateMachine() {
  switch (currentState) {

    case STATE_IDLE:
      // Do nothing — servo stays at current position
      break;

    case STATE_SCANNING: {
      // Sweep left ↔ right smoothly, 1° at a time
      unsigned long now = millis();
      if (now - lastStepTime >= (unsigned long)SCAN_STEP_DELAY) {
        lastStepTime = now;

        currentAngle += scanDirection;

        // Reverse direction at boundaries
        if (currentAngle >= SERVO_MAX_ANGLE) {
          currentAngle = SERVO_MAX_ANGLE;
          scanDirection = -1;
        } else if (currentAngle <= SERVO_MIN_ANGLE) {
          currentAngle = SERVO_MIN_ANGLE;
          scanDirection = 1;
        }

        servo.write((int)currentAngle);
      }
      break;
    }

    case STATE_TRACKING: {
      // Proportional smooth movement toward target angle.
      // Step size is proportional to distance — big gap = faster move,
      // small gap = tiny nudge. This eliminates jerkiness.
      unsigned long now = millis();
      if (now - lastStepTime >= (unsigned long)TRACK_STEP_DELAY) {
        lastStepTime = now;

        float error = (float)targetAngle - currentAngle;
        float absError = abs(error);

        if (absError < TRACK_MIN_STEP) {
          // Close enough — snap to target and stop moving
          currentAngle = targetAngle;
        } else {
          // Proportional step: move 25% of remaining distance per tick,
          // clamped between MIN_STEP and MAX_STEP
          float step = absError * 0.25;
          step = constrain(step, TRACK_MIN_STEP, TRACK_MAX_STEP);

          if (error > 0) {
            currentAngle += step;
          } else {
            currentAngle -= step;
          }
        }

        // Clamp to valid range
        currentAngle = constrain(currentAngle, (float)SERVO_MIN_ANGLE, (float)SERVO_MAX_ANGLE);
        servo.write((int)currentAngle);
      }
      break;
    }
  }
}

// ============================================================
//  STATE TRANSITIONS
// ============================================================
void setState(State newState) {
  if (newState == currentState) return;

  currentState = newState;

  switch (newState) {
    case STATE_IDLE:
      Serial.println("STATE:IDLE");
      break;

    case STATE_SCANNING:
      // Determine initial scan direction from current position
      if (currentAngle >= SERVO_MAX_ANGLE - 5) {
        scanDirection = -1;
      } else if (currentAngle <= SERVO_MIN_ANGLE + 5) {
        scanDirection = 1;
      }
      // else keep current direction for smooth transition
      Serial.println("STATE:SCANNING");
      break;

    case STATE_TRACKING:
      Serial.println("STATE:TRACKING");
      break;
  }
}

// ============================================================
//  ANGLE REPORTING
// ============================================================
void reportAngle() {
  Serial.print("ANGLE:");
  Serial.println((int)currentAngle);
}
