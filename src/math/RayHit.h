#ifndef RAYHIT_H
#define RAYHIT_H
#endif

#include "Vector3.h"
#include "Ray.h"
#include "Solid.h"

class RayHit {
    private:
    Solid hitSolid;
    Ray ray;
    Vector3 hitPos;
    Vector3 normal;

    public:
        RayHit(Ray r, Solid hs, Vector3 hp) {
            ray = r;
            hitSolid = hs;
            hitPos = hp; 
            normal = hitSolid.getNormalAt(hitPos);
        }
        Ray getRay() {return ray; };
        Solid getSolid() {return hitSolid; };
        Vector3 getPosition() {return hitPos; };
        Vector3 getNormal() {return normal; };
};

