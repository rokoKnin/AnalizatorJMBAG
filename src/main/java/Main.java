import Image.Image;
import Network.NeuralNetwork;
import Utils.ImageProcessorUtils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        String number = "0001534247";
        ArrayList<BufferedImage> images = new ArrayList<>();

        String inputFilePath = "src/main/resources/JMBAG_dataSet/trainingPictures/0001534247.jpeg";
        String outputOtsuFilePath = "src/main/resources/testPictures/grayscale/" + number + "_binarized";
        String inputNeuralNetwork = "src\\main\\resources\\MNIST_nn\\NeuralNetwork_1.txt";
        try {
            BufferedImage originalImage = ImageIO.read(new File(inputFilePath));
            if (originalImage == null) {
                System.out.println("Error: Could not load the image.");
                return;
            }

            BufferedImage grayImage = ImageProcessorUtils.processImage(originalImage);

            List<Image> numbers = ImageProcessorUtils.segmentNumbers(grayImage, number.split(""));


            FileReader fileReader = new FileReader(inputNeuralNetwork);
            NeuralNetwork nn = NeuralNetwork.fromString(fileReader.readAllAsString());

            int i = 0, result = 0;
            for (Image image : numbers.reversed()) {
                result += (int) (Math.pow(10, i) * nn.guess(image));
            }
            System.out.println("I think the number is: " + result);
            System.out.println("The real number is: " + number);
        } catch (IOException e) {
            System.out.println("An error occurred: Line 42\n" + e.getMessage());
        }
    }
}
