#ifndef VECTOR3_H
#define VECTOR3_H
#endif

#include <cmath>

#define degreesToRadians(angleDegrees) ((angleDegrees) * M_PI / 180.0)
#define radiansToDegrees(angleRadians) ((angleRadians) * 180.0 / M_PI)

class Vector3 {
    private:
        float x, y, z;
    public:
        Vector3(float xx, float yy, float zz);
        Vector3();
        Vector3(float xx);

        float getX() {return x; };
        float getY() {return y; };
        float getZ() {return z; };

        void setX(float xx) { x = xx; };
        void setY(float yy) { y = yy; };
        void setZ(float zz) { z = zz; };

        Vector3 add(Vector3 vec);
        Vector3 subtract(Vector3 vec);
        Vector3 multiply(float scalar);
        Vector3 multiply(Vector3 vec);
        Vector3 divide (Vector3 vec);

        float length();
        Vector3 normalize();
        Vector3 rotateYP(float yaw, float pitch);
        void translate(Vector3 vec);
        static float distance(Vector3 a, Vector3 b);
        static float dot(Vector3 a, Vector3 b);
        static Vector3 lerp(Vector3 a, Vector3 b, float t);
        Vector3 clone();
};