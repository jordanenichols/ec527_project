#include "Vector3.h"

Vector3::Vector3(float xx, float yy, float zz) {
    x = xx;
    y = yy;
    z = zz;
}
Vector3::Vector3() {
    x = 0;
    y = 0;
    z = 0;
}
Vector3::Vector3(float xx) {
    x = xx;
    y = xx;
    z = xx;
}
Vector3 Vector3::add(Vector3 vec) {
    return Vector3(x + vec.x, y + vec.y, z + vec.z);
}

Vector3 Vector3::subtract(Vector3 vec) {
    return Vector3(x - vec.x, y - vec.y, z - vec.z);
}

Vector3 Vector3::multiply(float scalar) {
    return Vector3(x * scalar, y * scalar, z * scalar);
}


Vector3 Vector3::multiply(Vector3 vec) {
    return Vector3(x * vec.x, y * vec.y, z * vec.z);
}

Vector3 Vector3::divide(Vector3 vec) {
    return Vector3(x / vec.x, y / vec.y, z / vec.z);
}

float Vector3::length() {
    return sqrt(x*x+y*y+z*z);
}

Vector3 Vector3::normalize() {
    float len = length();
    return Vector3(x / len, y / len, z / len);

}

Vector3 Vector3::rotateYP(float yaw, float pitch) {
    float yawRads = degreesToRadians(yaw);
    float pitchRads = degreesToRadians(pitch);

    /* Rotate around X axis (pitch) */
    float yy = (float) (y*cos(pitchRads) - z*sin(pitchRads));
    float zz = (float) (y*sin(pitchRads) - z*cos(pitchRads));

    /* Rotate around Y axis (yaw) */
    float xx = (float) (x*cos(yawRads) + zz*sin(yawRads));
    zz = (float) (-x*sin(yawRads) + zz*cos(yawRads));

    return Vector3(xx, yy, zz);
}

void Vector3::translate(Vector3 vec) {
    x += vec.x;
    y += vec.y;
    z += vec.z;
}

float Vector3::distance(Vector3 a, Vector3 b) {
    return (float) sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

float Vector3::dot(Vector3 a, Vector3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector3 Vector3::lerp(Vector3 a, Vector3 b, float t) {
    return a.add(b.subtract(a).multiply(t));
}

Vector3 Vector3::clone() {
    return Vector3(x, y, z);
}
