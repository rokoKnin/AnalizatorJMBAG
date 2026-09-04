package Utils;

import Image.Image;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class ImageProcessorUtils {

    private static final int CANVAS_SIZE = 28;      // Target dimensions: 32x32
    private static final int MAX_DIGIT_SIZE = 22;   // Max digit size (leaves padding on all sides)

    /**
     * Resizes a list of segmented images to 32x32 grayscale images with anti-aliasing (gray edges)
     * and a 2px safety padding on all sides.
     *
     * @param originalImages List of segmented digit BufferedImages
     * @return List of 32x32 grayscale BufferedImages with preserved gray gradients
     */
    public static List<BufferedImage> formatTo32x32(List<BufferedImage> originalImages) {
        List<BufferedImage> formattedImages = new ArrayList<>();

        for (BufferedImage src : originalImages) {
            // 1. Create a 32x32 grayscale canvas initialized to solid black
            BufferedImage canvas = new BufferedImage(CANVAS_SIZE, CANVAS_SIZE, BufferedImage.TYPE_BYTE_GRAY);
            Graphics2D g = canvas.createGraphics();

            // Fill canvas with black
            g.setColor(Color.BLACK);
            g.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

            // Enable Bilinear Interpolation to smoothly calculate gray anti-aliased pixels
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

            // Enable general anti-aliasing for smoother stroke rendering
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // 2. Calculate aspect-ratio scaling to fit within the 28x28 inner box
            int srcWidth = src.getWidth();
            int srcHeight = src.getHeight();

            double scale = Math.min((double) MAX_DIGIT_SIZE / srcWidth, (double) MAX_DIGIT_SIZE / srcHeight);
            int newWidth = (int) Math.round(srcWidth * scale);
            int newHeight = (int) Math.round(srcHeight * scale);

            // 3. Center the digit inside the 32x32 canvas
            int x = (CANVAS_SIZE - newWidth) / 2;
            int y = (CANVAS_SIZE - newHeight) / 2;

            // 4. Draw the image onto the canvas
            g.drawImage(src, x, y, newWidth, newHeight, null);
            g.dispose();

            formattedImages.add(canvas);
        }

        return formattedImages;
    }

    // Threshold above which a pixel is considered part of a digit (0 to 255)
    private static final int PIXEL_THRESHOLD = 100;

    // Ignore any component with fewer pixels than this to eliminate small noise spots
    private static final int MIN_PIXEL_AREA = 25;

    // Maximum gap (in pixels) between disconnected strokes of the same digit to be merged
    private static final int MERGE_DISTANCE = 12;

    /**
     * Takes an image with white numbers on a black background and extracts each digit into its own image.
     *
     * @param inputImage The input grayscale BufferedImage
     * @return List of BufferedImage objects, each containing a single segmented number
     */
    public static List<BufferedImage> segmentNumbers(BufferedImage inputImage) {
        int width = inputImage.getWidth();
        int height = inputImage.getHeight();

        boolean[][] visited = new boolean[width][height];
        List<Rectangle> detectedBoxes = new ArrayList<>();

        // Step 1: Find all connected white pixel components using BFS
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (!visited[x][y] && isForegroundPixel(inputImage, x, y)) {
                    Rectangle box = exploreComponent(inputImage, visited, x, y);

                    // Filter out small noise artifacts
                    if (box != null && (box.width * box.height) >= MIN_PIXEL_AREA) {
                        detectedBoxes.add(box);
                    }
                }
            }
        }

        // Step 2: Merge bounding boxes that are close together (e.g. broken strokes or multi-part digits)
        List<Rectangle> mergedBoxes = mergeCloseBoxes(detectedBoxes, MERGE_DISTANCE);

        // Step 3: Crop sub-images for each final bounding box
        List<BufferedImage> digitImages = new ArrayList<>();
        for (Rectangle box : mergedBoxes) {
            BufferedImage subImage = inputImage.getSubimage(box.x, box.y, box.width, box.height);

            // Create a deep copy so subImage isn't bound to the original memory buffer
            BufferedImage copy = new BufferedImage(box.width, box.height, BufferedImage.TYPE_BYTE_GRAY);
            copy.getGraphics().drawImage(subImage, 0, 0, null);
            digitImages.add(copy);
        }

        return digitImages;
    }

    /**
     * Performs a Breadth-First Search (BFS) to group connected foreground pixels.
     */
    private static Rectangle exploreComponent(BufferedImage img, boolean[][] visited, int startX, int startY) {
        int width = img.getWidth();
        int height = img.getHeight();

        int minX = startX, maxX = startX;
        int minY = startY, maxY = startY;

        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{startX, startY});
        visited[startX][startY] = true;

        // 8-directional neighbor offsets to bridge thin diagonal connections
        int[] dx = {-1, 0, 1, -1, 1, -1, 0, 1};
        int[] dy = {-1, -1, -1, 0, 0, 1, 1, 1};

        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            int px = point[0];
            int py = point[1];

            minX = Math.min(minX, px);
            maxX = Math.max(maxX, px);
            minY = Math.min(minY, py);
            maxY = Math.max(maxY, py);

            for (int i = 0; i < 8; i++) {
                int nx = px + dx[i];
                int ny = py + dy[i];

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    if (!visited[nx][ny] && isForegroundPixel(img, nx, ny)) {
                        visited[nx][ny] = true;
                        queue.add(new int[]{nx, ny});
                    }
                }
            }
        }

        return new Rectangle(minX, minY, (maxX - minX + 1), (maxY - minY + 1));
    }

    /**
     * Merges bounding boxes that lie within a specified pixel proximity threshold.
     */
    private static List<Rectangle> mergeCloseBoxes(List<Rectangle> boxes, int distanceThreshold) {
        boolean merged = true;
        List<Rectangle> currentBoxes = new ArrayList<>(boxes);

        while (merged) {
            merged = false;
            List<Rectangle> nextBoxes = new ArrayList<>();

            while (!currentBoxes.isEmpty()) {
                Rectangle b1 = currentBoxes.remove(0);
                boolean combined = false;

                for (int i = 0; i < currentBoxes.size(); i++) {
                    Rectangle b2 = currentBoxes.get(i);

                    if (shouldMerge(b1, b2, distanceThreshold)) {
                        // Expand b1 to encompass b2
                        Rectangle newBox = b1.union(b2);
                        currentBoxes.set(i, newBox);
                        combined = true;
                        merged = true;
                        break;
                    }
                }

                if (!combined) {
                    nextBoxes.add(b1);
                }
            }
            currentBoxes = nextBoxes;
        }

        return currentBoxes;
    }

    /**
     * Determines whether two bounding boxes are close enough horizontally and vertically to belong to the same digit.
     */
    private static boolean shouldMerge(Rectangle r1, Rectangle r2, int margin) {
        Rectangle expandedR1 = new Rectangle(
                r1.x - margin,
                r1.y - margin,
                r1.width + (2 * margin),
                r1.height + (2 * margin)
        );
        return expandedR1.intersects(r2);
    }

    /**
     * Checks if a pixel at (x, y) is white/bright enough to be considered foreground.
     */
    private static boolean isForegroundPixel(BufferedImage img, int x, int y) {
        int rgb = img.getRGB(x, y);
        int gray = rgb & 0xFF; // Extracts lower 8 bits for grayscale intensity
        return gray > PIXEL_THRESHOLD;
    }

    public static BufferedImage processImage(BufferedImage originalImage) {
        int width = originalImage.getWidth();
        int height = originalImage.getHeight();
        int totalPixels = width * height;

        // Step A: Convert to Grayscale
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

        // Step B: Apply 3x3 Gaussian Blur to remove high-frequency noise
//        int[][] blurredMatrix = applyGaussianBlur3x3(grayMatrix, width, height);
        int[][] blurredMatrix = applyFastBoxBlur(grayMatrix, width, height, 5, 3);

        // Step C: Generate histogram using the blurred matrix
        int[] histData = new int[256];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                histData[blurredMatrix[x][y]]++;
            }
        }

        // Step D: Calculate Otsu's optimal threshold value
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

        // Step E: Apply threshold to create final binarized image
        BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int binaryVal = (blurredMatrix[x][y] > threshold) ? 0 : 255;
                grayImage.getRaster().setSample(x, y, 0, binaryVal);
            }
        }

        return grayImage;
    }

    private static int[][] applyGaussianBlur3x3(int[][] input, int width, int height) {
        int[][] output = new int[width][height];

        int[] kernel = {
                1, 4, 7, 4, 1,
                4, 16, 26, 16, 4,
                7, 26, 41, 26, 7,
                4, 16, 26, 16, 4,
                1, 4, 7, 4, 1
        };
        int kernelWeight = 273;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int sum = 0;
                int kIdx = 0;

                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        // Clamp boundary pixels to prevent border artifacts
                        int px = Math.min(Math.max(x + dx, 0), width - 1);
                        int py = Math.min(Math.max(y + dy, 0), height - 1);

                        sum += input[px][py] * kernel[kIdx++];
                    }
                }
                output[x][y] = sum / kernelWeight;
            }
        }
        return output;
    }

    public static int[][] applyFastBoxBlur(int[][] input, int width, int height, int radius, int passes) {
        int[][] current = input;

        for (int p = 0; p < passes; p++) {
            int[][] outputHorizontal = new int[width][height];
            int[][] outputVertical = new int[width][height];

            // Pass 1: Horizontal Blur O(1)
            blurHorizontal(current, outputHorizontal, width, height, radius);

            // Pass 2: Vertical Blur O(1)
            blurVertical(outputHorizontal, outputVertical, width, height, radius);

            current = outputVertical;
        }

        return current;
    }

    private static void blurHorizontal(int[][] in, int[][] out, int width, int height, int radius) {
        int windowSize = radius * 2 + 1;

        for (int y = 0; y < height; y++) {
            int sum = 0;

            // Initialize the accumulator for the leftmost window with boundary clamping
            for (int x = -radius; x <= radius; x++) {
                int px = Math.min(Math.max(x, 0), width - 1);
                sum += in[px][y];
            }

            // Slide window horizontally across row: O(1) per pixel
            for (int x = 0; x < width; x++) {
                out[x][y] = sum / windowSize;

                // Subtract left pixel leaving the window, add right pixel entering the window
                int leftPx = Math.min(Math.max(x - radius, 0), width - 1);
                int rightPx = Math.min(Math.max(x + radius + 1, 0), width - 1);

                sum += in[rightPx][y] - in[leftPx][y];
            }
        }
    }

    private static void blurVertical(int[][] in, int[][] out, int width, int height, int radius) {
        int windowSize = radius * 2 + 1;

        for (int x = 0; x < width; x++) {
            int sum = 0;

            // Initialize the accumulator for the topmost window with boundary clamping
            for (int y = -radius; y <= radius; y++) {
                int py = Math.min(Math.max(y, 0), height - 1);
                sum += in[x][py];
            }

            // Slide window vertically down column: O(1) per pixel
            for (int y = 0; y < height; y++) {
                out[x][y] = sum / windowSize;

                // Subtract top pixel leaving the window, add bottom pixel entering the window
                int topPy = Math.min(Math.max(y - radius, 0), height - 1);
                int bottomPy = Math.min(Math.max(y + radius + 1, 0), height - 1);

                sum += in[x][bottomPy] - in[x][topPy];
            }
        }
    }

    public static Image convertToImage(BufferedImage canvas, int label) {
        int width = canvas.getWidth();
        int height = canvas.getHeight();
        double[][] data = new double[height][width];

        for (int row = 0; row < height; row++) {
            for (int column = 0; column < width; column++) {
                data[row][column] = canvas.getRaster().getSample(column, row, 0);
            }
        }

        return new Image(data, label);
    }
}