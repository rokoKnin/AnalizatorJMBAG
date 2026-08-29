import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.Buffer;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        ArrayList<BufferedImage> images = new ArrayList<>();

        String inputFilePath = "src/main/resources/testPictures/12378900.png";
        String outputFixedFilePath = "src/main/resources/testPictures/grayscale/12378900_fixed.png";
        String outputOtsuFilePath = "src/main/resources/testPictures/grayscale/12378900_otsu.png";
        String outputAdaptiveFilePath = "src/main/resources/testPictures/grayscale/12378900_adaptive.png";

        try {
            BufferedImage originalImage = ImageIO.read(new File(inputFilePath));
            if (originalImage == null) {
                System.out.println("Error: Could not load the image.");
                return;
            }

//            BufferedImage grayImage = fixed(originalImage);
            BufferedImage grayImage = otsu(originalImage);
//            BufferedImage grayImage = adaptive(originalImage);

            File outputFile = new File(outputOtsuFilePath);
            outputFile.getParentFile().mkdirs();
            ImageIO.write(grayImage, "png", outputFile);
            System.out.println("Otsu binarization complete!");

        } catch (IOException e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
    }

    public static BufferedImage fixed(BufferedImage originalImage) {
        int width = originalImage.getWidth();
        int height = originalImage.getHeight();
        int threshold = 100; // Adjustable cutoff value (0 to 255)

        // True single-channel grayscale image (1 byte per pixel)
        BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = originalImage.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                int gray = (int) (0.299 * r + 0.587 * g + 0.114 * b);

                // Binarization step
                int binaryVal = (gray > threshold) ? 0 :  255;

                // Set pixel value directly in the grayscale raster
                grayImage.getRaster().setSample(x, y, 0, binaryVal);
            }
        }

        return grayImage;

    }

    public static BufferedImage otsu(BufferedImage originalImage) {
        int width = originalImage.getWidth();
        int height = originalImage.getHeight();
        int totalPixels = width * height;

        // Step A: Generate histogram and cache grayscale values to avoid recalculating
        int[] histData = new int[256];
        int[][] grayMatrix = new int[width][height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = originalImage.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int gray = (int) (0.299 * r + 0.587 * g + 0.114 * b);

                grayMatrix[x][y] = gray;
                histData[gray]++;
            }
        }

        // Step B: Calculate Otsu's optimal threshold value
        float sum = 0;
        for (int t = 0; t < 256; t++) {
            sum += t * histData[t];
        }

        float sumB = 0;
        int wB = 0;
        float varMax = 0;
        int threshold = 0;

        for (int t = 0; t < 256; t++) {
            wB += histData[t];
            if (wB == 0) continue;
            int wF = totalPixels - wB;
            if (wF == 0) break;

            sumB += (float) (t * histData[t]);
            float mB = sumB / wB;
            float mF = (sum - sumB) / wF;

            float varBetween = (float) wB * (float) wF * (mB - mF) * (mB - mF);
            if (varBetween > varMax) {
                varMax = varBetween;
                threshold = t;
            }
        }
        System.out.println("Otsu calculated optimal threshold: " + threshold);

        // Step C: Apply threshold using TYPE_BYTE_GRAY
        BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int binaryVal = (grayMatrix[x][y] > threshold) ? 0 : 255;
                grayImage.getRaster().setSample(x, y, 0, binaryVal);
            }
        }

        return grayImage;

    }

    public static BufferedImage adaptive(BufferedImage originalImage) {
        int width = originalImage.getWidth();
        int height = originalImage.getHeight();

        // 1. Convert and cache to a 2D grayscale array
        int[][] grayMatrix = new int[width][height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = originalImage.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                grayMatrix[x][y] = (int) (0.299 * r + 0.587 * g + 0.114 * b);
            }
        }

        BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        // 2. Parameters for local neighborhood block
        int S = 15;     // Block size (must be odd)
        double C = 4.0; // Constant offset

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                int x1 = Math.max(0, x - S / 2);
                int x2 = Math.min(width - 1, x + S / 2);
                int y1 = Math.max(0, y - S / 2);
                int y2 = Math.min(height - 1, y + S / 2);

                int count = 0;
                long sum = 0;

                for (int dy = y1; dy <= y2; dy++) {
                    for (int dx = x1; dx <= x2; dx++) {
                        sum += grayMatrix[dx][dy];
                        count++;
                    }
                }

                double localThreshold = (sum / (double) count) - C;
                int binaryVal = (grayMatrix[x][y] > localThreshold) ? 0 : 255;

                grayImage.getRaster().setSample(x, y, 0, binaryVal);
            }
        }

        return grayImage;

    }
}
