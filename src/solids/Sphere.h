#ifndef SPHERE_H
#define SPHERE_H
#endif

#include "Solid.h"
#include <cmath>

class Sphere : public Solid {
    private:
        friend class Solid;
        float radius;

    public:
        Sphere(Vector3 p, float r, Color c, float rad, float e) : Solid(p, c, r, e) {
            radius = rad;
        }
        Vector3 calculateIntersection(Ray ray) {
            float t = Vector3::dot(getPosition().subtract(ray.getOrigin()), ray.getDirection());
            Vector3 p = ray.getOrigin().add(ray.getDirection().multiply(t));

            float y = getPosition().subtract(p).length();
            if(y < radius) {
                float x = (float) sqrt(radius*radius - y*y);
                float t1 = t-x;
                if(t1 > 0) return ray.getOrigin().add(ray.getDirection().multiply(t1));
                else return Vector3(0, 0, 0); 
            } else {
                return Vector3(0, 0, 0); 
            }
        }

        Vector3 getNormalAt(Vector3 point) {
            return point.subtract(getPosition()).normalize();
        }
};