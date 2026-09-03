package MNIST;

import Network.NetworkBuilder;
import Network.NeuralNetwork;

import javax.xml.crypto.Data;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import static MNIST.ExecutorServiceTester.runParallelTest;
import static java.util.Collections.shuffle;

public class mainMNIST {
    public static void main(String[] args) throws IOException {
        String ID = "NeuralNetwork_1";
        long SEED = 3456;

        System.out.println("Starting MNIST data Loading...");

        List<Image> imagesTest = new DataReader().readData("src/main/resources/MNIST_dataSet/mnist_test.csv");
        List<Image> imagesTrain = new DataReader().readData("src/main/resources/MNIST_dataSet/mnist_train.csv");

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test Size: " + imagesTest.size());

        NetworkBuilder nb = new NetworkBuilder(ID, 28, 28, 255);
        nb.addConvolutionLayer(8,5,1,0.001,SEED);
        nb.addMaxPoolLayer(3,2);
        nb.addFullyConnectedLayer(10, 0.001, SEED);

        NeuralNetwork nn = nb.build();

        float rate = runParallelTest(nn, imagesTest, 20);
        System.out.println("Pre-training test success rate: " + rate);

        int epochs = 10;

        for (int i = 0; i < epochs; i++) {
        shuffle(imagesTrain);
        nn.train(imagesTrain);

        rate = runParallelTest(nn, imagesTest, 20);
        System.out.println("test success rate at epoch " + i + ": " + rate);
        }

        System.out.println();
        System.out.println(nn);

        String outputFilePath = "src/main/resources/MNIST_nn/" + ID + ".txt";
        FileWriter outputFile = new FileWriter(outputFilePath);
        outputFile.write(nn.toString());
    }
}
