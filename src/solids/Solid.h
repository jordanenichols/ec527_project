#ifndef SOLID_H
#define SOLID_H
#endif

#include "../pixeldata/Color.h"
#include "../math/Ray.h"

class Solid {
    private:
        Vector3 position;
        Color color;
        float reflectivity;
        float emission;
    public:
        Solid(Vector3 p, Color c, float r, float e) {
            position = p;
            color = c;
            reflectivity = r;
            emission = e;
        }
        Vector3 getPosition() { return position; };
        Color getColor() {return color; };
        Color getTextureColor(Vector3 point) { return getColor(); };
        float getReflectivity() {return reflectivity; };
        float getEmission() {return emission; };

        Vector3 calculateIntersection(Ray ray);
        Vector3 getNormalAt(Vector3 point);
};