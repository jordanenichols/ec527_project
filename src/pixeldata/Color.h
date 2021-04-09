#ifndef COLOR_H
#define COLOR_H
#endif

#include <cmath>

class Color {
    private:
    float red, green, blue;

    public:
        Color(float r, float g, float b);
        Color();
        float getRed() {return red; };
        float getGreen() {return green; };
        float getBlue() {return blue; };

        Color multiply(Color other);
        Color multiply(float brightness);
        Color add(Color other);
        void addSelf(Color other);
        Color add(float brightness);
        int getRGB();
        float getLuminance();
        static Color fromInt(int argb);
        static Color average(Color *colors);
        static Color average(Color *colors, float *weights);
        static float lerp(float a, float b, float t);
        static Color lerp(Color a, Color b, float t);


};

        static Color BLACK = Color(0, 0, 0);
        static Color WHITE = Color(1, 1, 1);
        static Color RED = Color(1, 0, 0);
        static Color GREEN = Color(0, 1, 0);
        static Color BLUE = Color(0, 0, 1);
        static Color MAGENTA = Color(1, 0, 1);
        static Color GRAY = Color(0.5, 0.5, 0.5);
        static Color DARK_GRAY = Color(0.2, 0.2, 0.2);