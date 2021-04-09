#pragma once
#ifndef LINE_H
#define LINE_H
#endif

#include "Vector3.h"
#include "Ray.h"

class Line {
    public:
        Vector3 pointA;
        Vector3 pointB;

        Line(Vector3 pointAA, Vector3 pointBB) {
            pointA = pointAA;
            pointB = pointBB;
        }
        
        Ray asRay() {
            return Ray(pointA, pointB.subtract(pointA).normalize());
        }
};