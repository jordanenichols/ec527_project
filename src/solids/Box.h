#pragma once

#ifndef BOX_H
#define BOX_H
#endif

#include "../Prerequisites.h"

class Box : public Solid {
    private:
        Vector3 min, max;
    
    public:
        Box(Vector3 position, Vector3 scale, Color c, float r, float e) : Solid(position, c, r, e){
            max = position.add(scale.multiply(0.5));
            min = position.subtract(scale.multiply(0.5));
        }
private:
    friend class Solid;
};