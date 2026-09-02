import Utils.ImageProcessorUtils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        String number = "0118032914";
        ArrayList<BufferedImage> images = new ArrayList<>();

        String inputFilePath = "src/main/resources/testPictures/" + number + ".jpeg";
        String outputOtsuFilePath = "src/main/resources/testPictures/grayscale/" + number + "_binarized";
        try {
            BufferedImage originalImage = ImageIO.read(new File(inputFilePath));
            if (originalImage == null) {
                System.out.println("Error: Could not load the image.");
                return;
            }

            BufferedImage grayImage = ImageProcessorUtils.processImage(originalImage);

            List<BufferedImage> numbers = ImageProcessorUtils.segmentNumbers(grayImage);

            numbers = ImageProcessorUtils.formatTo32x32(numbers);

            for (BufferedImage image : numbers) {
                File outputFile = new File(outputOtsuFilePath + "_" + image.hashCode() + ".png");
                outputFile.getParentFile().mkdirs();
                ImageIO.write(image, "png", outputFile);
                System.out.println("Exported image");
            }

        } catch (IOException e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
    }
}
