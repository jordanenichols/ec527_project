#include "PixelData.h"
#include "Color.h"
#include <stdio.h>

class PixelBuffer
{
private:
    PixelData[][] pixels;
    int width, height;

public:
    PixelBuffer(int wwidth, hheight)
    {
        width = wwidth;
        height = hheight;
        pixels = new PixelData[width][height];
    }
    void setPixel(int x, int y, PixelData pixelData) {
        pixels[x][y] = pixelData;
    }
    PixelData getPixel(int x, int y) {
        return pixels[x][y];
    }

    void filterByEmission(float minEmission) {
        for (int i = 0; i<pixels.length; i++) {
            for (int j = 0; j<pixels[i].length; j++) {
                PixelData pxl = pixels[i][j];
                if (pxl != null && pxl.getEmission() < minEmission) {
                    pixels[i][j] = new PixelData(Color.BLACK, pxl.getDepth(), pxl.getEmission());
                }
            }
        }
    }

    /** Changes will be applied to the buffer itself */
    PixelBuffer add(PixelBuffer other) {
        for (int i = 0; i<pixels.length; i++) {
            for (int j = 0; j<pixels[i].length; j++) {
                PixelData pxl = pixels[i][j];
                PixelData otherPxl = other.pixels[i][j];
                if (pxl != null && otherPxl != null) {
                    float brightnessB4 = pixels[i][j].getColor().getLuminance();
                    pixels[i][j].add(otherPxl);
                }
            }
        }
        return *this;
    }
    /** Changes will be applied to the buffer itself */
    PixelBuffer multiply(float brightness) {
        for (int i = 0; i<pixels.length; i++) {
            for (int j = 0; j<pixels[i].length; j++) {
                pixels[i][j].multiply(brightness);
            }
        }
        return *this;
    }

    PixelBuffer resize(int newWidth, int newHeight, boolean linear) { // Linear resizing isn't actually implemented yet.
        PixelBuffer copy = new PixelBuffer(newWidth, newHeight);
        for (int i = 0; i<newWidth; i++) {
            for (int j = 0; j<newHeight; j++) {
                copy.pixels[i][j] = pixels[(int)((float)i/newWidth*width)][(int)((float)j/newHeight*height)];
            }
        }
        return copy;
    }

    int getWidth() {
        return width;
    }

    int getHeight() {
        return height;
    }

    void countEmptyPixels() {
        int emptyPixels = 0;
        for (int i = 0; i < pixels.length; i++) {
            
            //          this was null in java!!!
            if (pixels[i] == nullptr) emptyPixels++;

        }
        std::cout << "Found " << emptyPixels << " empty pixels." << std::endl;
    }

    PixelBuffer clone() {
        PixelBuffer clone = new PixelBuffer(width, height);
        for (int i = 0; i < pixels.length; i++) {

        //  System.arraycopy(pixels[i], 0, clone.pixels[i], 0, pixels[i].length);
            memmove(clone.pixels, pixels[i], pixels[i].length);

        }
        return clone;
    }
};