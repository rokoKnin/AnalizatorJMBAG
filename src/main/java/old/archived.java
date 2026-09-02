package old;

import java.awt.image.BufferedImage;

public class archived {

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
