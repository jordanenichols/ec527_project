#include "PixelBuffer.h"
#include "Color.h"

class GaussianBlur
{
private:
    float[] kernel;
    PixelBuffer pixelBuffer;
    int width, height;

public:
    GaussianBlur(PixelBuffer ppixelBuffer)
    {
        pixelBuffer = ppixelBuffer;
        width = ppixelBuffer.getWidth();
        height = ppixelBuffer.getWidth();

        kernel = new float[]{0.0093F, 0.028002F, 0.065984F, 0.121703F, 0.175713F, 0.198596F, 0.175713F, 0.121703F, 0.065984F, 0.028002F, 0.0093F};
    }  

    blurHorizontally(int radius)
    {
        PixelBuffer result = new PixelBuffer(width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Color blurredColor = new Color(0, 0, 0);
                PixelData originalPixel = pixelBuffer.getPixel(x, y);
                for (int i = -radius; i <= radius; i++)
                {
                    float kernelMultiplier = kernel[(int)((i+radius)/(radius*2.0f)*(kernel.length-1))];
                    if (x+i>0 && x+i<width)
                    {
                        PixelData pixel = pixelBuffer.getPixel(x+i, y);
                        if (pixel != null)
                            blurredColor.addSelf(pixel.getColor().multiply(kernelMultiplier));
                    }
                }
                result.setPixel(x, y, new PixelData(blurredColor, originalPixel.getDepth(), originalPixel.getEmission()));
            }
        }
        pixelBuffer = result; // this.pixelBuffer = result; <-- origina java code??
    }
    
    void blurVertically(int radius) {
        PixelBuffer result = new PixelBuffer(width, height);
        for (int x = 0; x<width; x++) {
            for (int y = 0; y<height; y++) {
                Color blurredColor = new Color(0, 0, 0);
                PixelData originalPixel = pixelBuffer.getPixel(x, y);
                for (int i = -radius; i<=radius; i++) {
                    float kernelMultiplier = kernel[(int) ((i+radius)/(radius*2F)*(kernel.length-1))];
                    if (y+i>=0 && y+i<height) {
                        PixelData pixel = pixelBuffer.getPixel(x, y+i);
                        if (pixel != nullptr) // was originally null in java code!!!
                            blurredColor.addSelf(pixel.getColor().multiply(kernelMultiplier));
                    }
                }

                result.setPixel(x, y, new PixelData(blurredColor, originalPixel.getDepth(), originalPixel.getEmission()));
            }
        }
        pixelBuffer = result;
    }

    void blur(int radius, int iterations) {
        for (int i = 0; i<iterations; i++) {
            blurHorizontally(radius);
            blurVertically(radius);
        }
    }

    PixelBuffer getPixelBuffer() {
        return pixelBuffer;
    }

    // Currently unused due to kernel being hardcoded
    // float gaussianDistribution(float x, float sigma) {
    //     return (float) (1/Math.sqrt(2*Math.PI*sigma*sigma)*Math.exp(-(x*x)/(2*sigma*sigma))); // https://en.wikipedia.org/wiki/Gaussian_blur
    // }

};