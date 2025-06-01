#include <Wire.h>
#include <MPU6050.h>

// HC-SR04 Pin
const int trigPin = 9;
const int echoPin = 8;

// MPU6050
MPU6050 mpu;

// Variabel untuk waktu dan jarak
long duration;
float distance;

void setup() {
  // Serial komunikasi
  Serial.begin(9600);
  
  // HC-SR04 setup
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  // MPU6050 setup
  Wire.begin();
  mpu.initialize();

  // Cek koneksi MPU6050
  if (mpu.testConnection()) {
    Serial.println("MPU6050 terkoneksi.");
  } else {
    Serial.println("MPU6050 tidak terdeteksi!");
  }

  delay(1000);
}

void loop() {
  // Baca data dari HC-SR04
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;  // Konversi ke cm

  // Baca data dari MPU6050
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Tampilkan data ke Serial Monitor
  Serial.print("Jarak (cm): ");
  Serial.print(distance);
  Serial.print(" | Gyro (X,Y,Z): ");
  Serial.print(gx); Serial.print(", ");
  Serial.print(gy); Serial.print(", ");
  Serial.print(gz);
  Serial.println();

  delay(500);
}