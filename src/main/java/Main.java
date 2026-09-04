import Image.Image;
import Network.NetworkBuilder;
import Network.NeuralNetwork;
import Utils.ImageProcessorUtils;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static MNIST.ExecutorServiceTester.runParallelTest;
import static Utils.ImageProcessorUtils.convertToImage;
import static Utils.ImageProcessorUtils.formatTo32x32;
import static java.util.Collections.shuffle;


public class Main {
    public static void main(String[] args) {
        boolean training = false;
        boolean MNIST_nn = true;

        if (!training) {
            String networkPath = MNIST_nn
                    ? "src/main/resources/NeuralNetworks/NeuralNetwork_trained_MNIST.txt"
                    : "src/main/resources/NeuralNetworks/NeuralNetwork_trained.txt";

            try {
                // Load the previously trained network
                BufferedReader br = new BufferedReader(new FileReader(networkPath));
                NeuralNetwork nn = NeuralNetwork.fromString(br.readAllAsString());

                // Launch the UI on the Event Dispatch Thread
                SwingUtilities.invokeLater(() -> buildAndShowUI(nn));
            } catch (Exception e) {
                throw new RuntimeException("Error loading trained network: \n" + e.getMessage());
            }
        } else {
            String testPath = "Src/main/resources/JMBAG_dataSet/testPictures";
            String trainingPath = "Src/main/resources/JMBAG_dataSet/trainingPictures";

            try (Stream<Path> trainingStream = Files.walk(Paths.get(trainingPath));
                 Stream<Path> testStream = Files.walk(Paths.get(testPath));) {

                NeuralNetwork nn;

                if (MNIST_nn) {
                    String networkPath = "Src/main/resources/MNIST_nn/NeuralNetwork_1.txt";
                    BufferedReader br = new BufferedReader(new FileReader(networkPath));
                    nn = NeuralNetwork.fromString(br.readAllAsString());
                } else {
                    long SEED = 141134;

                    NetworkBuilder nb = new NetworkBuilder("NeuralNetwork_2.txt", 28, 28, 255);
                    nb.addConvolutionLayer(8, 5, 1, 0.001, SEED);
                    nb.addMaxPoolLayer(3, 2);
                    nb.addFullyConnectedLayer(10, 0.001, SEED);

                    nn = nb.build();
                }

                List<Image> trainingImages = java.util.Collections.synchronizedList(new ArrayList<>());
                List<Image> testImages = java.util.Collections.synchronizedList(new ArrayList<>());

                dataSerializer(trainingStream, trainingImages);
                dataSerializer(testStream, testImages);

                float rate = runParallelTest(nn, testImages, 20);
                System.out.println("Pre-training test success rate: " + rate);

                int epochs = 10;

                for (int i = 0; i < epochs; i++) {
                    shuffle(trainingImages);
                    nn.train(trainingImages);

                    rate = runParallelTest(nn, testImages, 20);
                    System.out.println("test success rate at epoch " + i + ": " + rate);
                }


                System.out.println();
                System.out.println(nn);

                String outputFilePath = MNIST_nn ? "src/main/resources/NeuralNetworks/NeuralNetwork_trained_MNIST.txt" : "src/main/resources/NeuralNetworks/NeuralNetwork_trained.txt";
                BufferedWriter bW = new BufferedWriter(new FileWriter(outputFilePath));
                bW.write(nn.toString());
            } catch (IOException e) {
                throw new RuntimeException("Error accured: \n" + e.getMessage());
            }
        }
    }

    private static void dataSerializer(Stream<Path> stream, List<Image> images) {
        stream.filter(Files::isRegularFile).parallel()
                .forEach(Path ->
        {
            try {
                BufferedImage originalImage = ImageIO.read(new File(Path.toUri()));
                BufferedImage grayImage = ImageProcessorUtils.processImage(originalImage);
                List<BufferedImage> numbers = ImageProcessorUtils.segmentNumbers(grayImage);
                numbers = formatTo32x32(numbers);

                String number = Path.getFileName().toString().split("\\.")[0];
                String[] numberArray = number.split("");

                for (int i = 0; i < numbers.size(); i++) {
                    // Safety check: Stop if segmentation found more parts than the filename has digits
                    if (i >= numberArray.length) {
                        System.err.println("Warning: Found extra image segments in file " + number + ". Skipping noise.");
                        break;
                    }

                    BufferedImage image = numbers.get(i);
                    Image netImage = convertToImage(image, Integer.parseInt(numberArray[i]));

                    // Remember to keep your thread-safe add block from the previous fix!
                    synchronized (images) {
                        images.add(netImage);
                    }
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    private static void buildAndShowUI(NeuralNetwork nn) {
        JFrame frame = new JFrame("CNN Number Recognizer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(500, 300);
        frame.setLayout(new BorderLayout(10, 10));

        // --- Top Panel: File Upload ---
        JPanel topPanel = new JPanel();
        JButton uploadBtn = new JButton("Upload Photo");
        JLabel fileLabel = new JLabel("No file selected");
        topPanel.add(uploadBtn);
        topPanel.add(fileLabel);

        // --- Center Panel: Inputs and Run ---
        JPanel centerPanel = new JPanel(new GridLayout(3, 1, 5, 5));

        JPanel inputPanel = new JPanel();
        inputPanel.add(new JLabel("Correct Answer (Expected):"));
        JTextField expectedAnswerField = new JTextField(15);
        inputPanel.add(expectedAnswerField);

        JPanel runPanel = new JPanel();
        JButton runBtn = new JButton("Run Prediction");
        runPanel.add(runBtn);

        JLabel resultLabel = new JLabel("CNN Prediction will appear here", SwingConstants.CENTER);
        resultLabel.setFont(new Font("Arial", Font.BOLD, 14));

        centerPanel.add(inputPanel);
        centerPanel.add(runPanel);
        centerPanel.add(resultLabel);

        frame.add(topPanel, BorderLayout.NORTH);
        frame.add(centerPanel, BorderLayout.CENTER);

        // --- Event Listeners ---
        final File[] selectedFile = {null};

        uploadBtn.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser("Src/main/resources/JMBAG_dataSet/testPictures");
            fileChooser.setFileFilter(new FileNameExtensionFilter("Images", "jpg", "png", "jpeg"));

            if (fileChooser.showOpenDialog(frame) == JFileChooser.APPROVE_OPTION) {
                selectedFile[0] = fileChooser.getSelectedFile();
                fileLabel.setText(selectedFile[0].getName());
            }
        });

        runBtn.addActionListener(e -> {
            if (selectedFile[0] == null) {
                JOptionPane.showMessageDialog(frame, "Please upload a photo first.", "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            try {
                // Process the single image similarly to dataSerializer
                BufferedImage originalImage = ImageIO.read(selectedFile[0]);
                BufferedImage grayImage = ImageProcessorUtils.processImage(originalImage);
                List<BufferedImage> numbers = ImageProcessorUtils.segmentNumbers(grayImage);
                numbers = formatTo32x32(numbers);

                StringBuilder cnnPrediction = new StringBuilder();

                for (BufferedImage image : numbers) {
                    // Pass 0 as a dummy label since we only need the image data for prediction
                    Image netImage = convertToImage(image, 0);

                    // Predict the digit. Adjust 'predict()' if your NeuralNetwork uses a different method name
                    int digit = nn.guess(netImage);
                    cnnPrediction.append(digit);
                }

                String expected = expectedAnswerField.getText().trim();
                String resultText = String.format("<html>Expected: %s<br>CNN Thought: <font color='%s'>%s</font></html>",
                        expected,
                        expected.equals(cnnPrediction.toString()) ? "green" : "red",
                        cnnPrediction.toString());

                resultLabel.setText(resultText);

            } catch (Exception ex) {
                JOptionPane.showMessageDialog(frame, "Error processing image: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }
        });

        frame.setLocationRelativeTo(null); // Center on screen
        frame.setVisible(true);
    }
}
