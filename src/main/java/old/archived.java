package old;

import Image.Image;
import Network.NeuralNetwork;
import Utils.ImageProcessorUtils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static Utils.ImageProcessorUtils.convertToImage;
import static Utils.ImageProcessorUtils.formatTo32x32;

public class archived {

    public static void main(String[] args) {
        String number = "0012592037";
        ArrayList<BufferedImage> images = new ArrayList<>();

        String inputFilePath = "src/main/resources/JMBAG_dataSet/trainingPictures/0012592037.jpeg";
        String outputOtsuFilePath = "src/main/resources/";
        String inputNeuralNetwork = "src\\main\\resources\\MNIST_nn\\NeuralNetwork_1.txt";
        try (BufferedReader br = new BufferedReader(new FileReader(inputNeuralNetwork))) {
            NeuralNetwork nn = NeuralNetwork.fromString(br.readAllAsString());
            BufferedImage originalImage = ImageIO.read(new File(inputFilePath));
            if (originalImage == null) {
                System.out.println("Error: Could not load the image.");
                return;
            }

            BufferedImage grayImage = ImageProcessorUtils.processImage(originalImage);

            File outputFiles = new File(outputOtsuFilePath + "_" + number + ".png");
            outputFiles.getParentFile().mkdirs();
            ImageIO.write(grayImage, "png", outputFiles);

            List<BufferedImage> numbers = ImageProcessorUtils.segmentNumbers(grayImage);

            numbers = formatTo32x32(numbers);

            StringBuilder sb = new StringBuilder();

            for (BufferedImage image : numbers) {
                File outputFile = new File(outputOtsuFilePath + numbers.indexOf(image) + ".png");
                outputFile.getParentFile().mkdirs();
                ImageIO.write(image, "png", outputFile);
                Image netImage = convertToImage(image, 1);
                int predictedDigit = nn.guess(netImage);
                sb.append(predictedDigit);
            }
            long result = Long.parseLong(sb.toString());
            System.out.println("I think the number is: " + result);
            System.out.println("The real number is: " + number);
        } catch (IOException e) {
            System.out.println("An error occurred: Line 42\n" + e.getMessage());
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
