#include "Color.h"

Color::Color(float r, float g, float b) {
    red = r;
    green = g;
    blue = b;
}

Color::Color() {
    red = 0;
    green = 0;
    blue = 0;
}

Color Color::multiply(Color other) {
    return Color(red * other.red, green * other.green, blue * other.blue);
}
Color Color::multiply(float brightness) {
    brightness = fmin(1, brightness);
    return Color(red * brightness, green * brightness, blue * brightness);
}

Color Color::add(Color other) {
    return Color(fmin(1, red + other.red), fmin(1, green + other.green), fmin(1, blue + other.blue));
}
void Color::addSelf(Color other) {
    red = fmin(1, red + other.red);
    green = fmin(1, green + other.green);
    blue = fmin(1, blue + other.blue);
}
Color Color::add(float brightness) {
    return Color(fmin(1, red + brightness), fmin(1, green + brightness), fmin(1, blue + brightness));
}
int Color::getRGB() {
    int redPart = (int)(red*255);
    int bluePart = (int)(blue*255);
    int greenPart = (int)(green*255);

    redPart = (redPart << 16) & 0x00FF0000;
    greenPart = (greenPart << 8) & 0x0000FF00;
    bluePart = bluePart & 0x000000FF;
    return 0xFF000000 | redPart | greenPart | bluePart;
}

float Color::getLuminance() {
    return red * 0.2126 + green * 0.7152 + blue * 0.0722;
}
Color Color::fromInt(int argb) {
    int b = (argb)&0xFF;
    int g = (argb>>8) & 0xFF;
    int r = (argb>>16) & 0xFF;

    return Color(((float)r)/255, ((float)g)/255, ((float)b)/255);
}
Color Color::average(Color *colors) {
    float rSum = 0;
    float gSum = 0;
    float bSum = 0;

    Color *iter = colors;
    int count = 0;
    
    while(iter) {
        rSum += iter->getRed();
        gSum += iter->getGreen();
        bSum += iter->getBlue();
        iter += sizeof(Color);
        count++;
    }
    return Color(rSum / count, gSum / count, bSum / count);
}

Color Color::average(Color *colors, float *weights) {
    float rSum = 0;
    float gSum = 0;
    float bSum = 0;

    Color *iter_c = colors;
    float *iter_f = weights;
    float weightSum = 0;
    int count = 0;
    
    while(iter_c) {
        rSum += iter_c->getRed() * *iter_f;
        gSum += iter_c->getGreen() * *iter_f;
        bSum += iter_c->getBlue() * *iter_f;
        iter_c += sizeof(Color);
        iter_f += sizeof(float);
        count++;
        weightSum += *iter_f;
    }
    return Color(rSum / weightSum, gSum / weightSum, bSum / weightSum); 
}
float Color::lerp(float a, float b, float t) {
    return a + t * (b - a);
}

Color Color::lerp(Color a, Color b, float t) {
    return Color(lerp(a.getRed(), b.getRed(), t), lerp(a.getGreen(), b.getGreen(), t), lerp(a.getBlue(), b.getBlue(), t));
}