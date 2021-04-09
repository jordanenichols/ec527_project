#pragma once
#ifndef RAY_H
#define RAY_H
#endif

#include "Vector3.h"
#include "Line.h"

class Ray {
    private:
        Vector3 origin;
        Vector3 direction;
        
    public:
        Ray(Vector3 o, Vector3 d) {
            origin = o;
            direction = d;
        }

        Line asLine(float length) {
            return Line(origin, origin.add(direction.multiply(length)));
        }

        Vector3 getOrigin() {
            return origin;
        }

        Vector3 getDirection() {
            return direction;
        }
};