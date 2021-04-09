#ifndef PIXELDATA_H
#define PIXELDATA_H
#endif

class PixelData {
    private:
        Color color;
        float depth;
        float emission;

    public:
        PixelData(Color c, float d, float e) {
            color = c;
            depth = d;
            emission = e;
        }
        Color getColor() {return color; };
        float getDepth() {return depth; };
        float getEmission() {return emission; };

        void add(PiexlData other) {
            color.addSelf(other.color);
            depth = (depth + other.depth) / 2;
            emission += other.emission;
        }

        void multiply(float brightness) {
            color = color.multiply(brightness);
        }
};