package Network;

import Layers.Layer;
import MNIST.Image;

import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;

import static Utils.MatrixUtils.add;
import static Utils.MatrixUtils.multiply;

public class NeuralNetwork {
    private String ID;
    private List<Layer> layers;
    private double scaleFactor;

    public NeuralNetwork(String ID, List<Layer> layers, double scaleFactor) {
        this.ID = ID;
        this.layers = layers;
        this.scaleFactor = scaleFactor;

        linkLayers();
    }

    public static NeuralNetwork fromString(String data) {
        Scanner scanner = new Scanner(data);
        if (!scanner.nextLine().equals("NEURAL_NETWORK")) return null;

        String ID = scanner.nextLine();
        double scaleFactor = Double.parseDouble(scanner.nextLine());
        NetworkBuilder builder = new NetworkBuilder(ID, 0, 0, scaleFactor);

        while (scanner.hasNextLine()) {
            String type = scanner.nextLine();

            if (type.equals("CONVOLUTION")) {
                String[] params = scanner.nextLine().split(",");
                int fSize = Integer.parseInt(params[0]);
                int sSize = Integer.parseInt(params[1]);
                int inLen = Integer.parseInt(params[2]);
                int inRow = Integer.parseInt(params[3]);
                int inCol = Integer.parseInt(params[4]);
                double lr = Double.parseDouble(params[5]);
                int numFilters = Integer.parseInt(params[6]);

                List<double[][]> filters = new ArrayList<>();
                for (int f = 0; f < numFilters; f++) {
                    String[] wStrings = scanner.nextLine().split(",");
                    double[][] filter = new double[fSize][fSize];
                    int idx = 0;
                    for (int i = 0; i < fSize; i++) {
                        for (int j = 0; j < fSize; j++) {
                            filter[i][j] = Double.parseDouble(wStrings[idx++]);
                        }
                    }
                    filters.add(filter);
                }
                builder.addPreTrainedConvolutionLayer(fSize, sSize, inLen, inRow, inCol, lr, filters);

            } else if (type.equals("MAXPOOL")) {
                String[] params = scanner.nextLine().split(",");
                builder.addPreTrainedMaxPoolLayer(
                        Integer.parseInt(params[0]), Integer.parseInt(params[1]),
                        Integer.parseInt(params[2]), Integer.parseInt(params[3]), Integer.parseInt(params[4])
                );
            } else if (type.equals("FULLY_CONNECTED")) {
                String[] params = scanner.nextLine().split(",");
                int _inLength = Integer.parseInt(params[0]);
                int _outLength = Integer.parseInt(params[1]);
                double _learningRate = Double.parseDouble(params[2]);
                long _SEED = Long.parseLong(params[3]);

                double[][] weights = new double[_inLength][_outLength];
                for (int i = 0; i < _inLength; i++) {
                    String[] weightsString = scanner.nextLine().split(",", -1);
                    for (int j = 0; j < _outLength; j++) {
                        weights[i][j] = Double.parseDouble(weightsString[j]);
                    }
                }

                double[] biases = new double[_outLength];
                String[] biasesString = scanner.nextLine().split(",");
                for (int i = 0; i < _outLength; i++) {
                    biases[i] = Double.parseDouble(biasesString[i]);
                }

                builder.addPreTrainedFullyConnectedLayer(_inLength,  _outLength, _learningRate, _SEED, weights, biases);
            }
        }
        scanner.close();
        return builder.build();
    }

    private void linkLayers() {
        if (layers.size() <= 1) return;

        layers.getFirst().set_nextLayer(layers.get(1));
        for(int i=1;i<layers.size() - 1;i++){
            layers.get(i).set_nextLayer(layers.get(i+1));
            layers.get(i).set_prevLayer(layers.get(i-1));
        }
        layers.getLast().set_prevLayer(layers.get(layers.size()-2));
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer){
        int numClasses = networkOutput.length;
        double[] expected = new double[numClasses];

        expected[correctAnswer] = 1;

        return add(networkOutput, multiply(expected, -1));
    }

    private int getMaxIndex(double[] in) {
        double max = Double.NEGATIVE_INFINITY;
        int index = -1;

        for(int i=0;i<in.length;i++){
            if(in[i] > max){
                max = in[i];
                index = i;
            }
        }
        return index;
    }

    public int guessMNIST(Image image) {
        List<double[][]> imList = new ArrayList<>();

        imList.add(multiply(image.getData(), 1.0/scaleFactor));

        double[] out = layers.getFirst().getOutput(imList);

        return getMaxIndex(out);
    }

    public float test (List<Image> images) {
        int correct = 0;

        for(Image image : images){
            int guess = guessMNIST(image);

            if (guess == image.getLabel()) {
                correct ++;
            }
        }

        return ((float)correct / images.size());
    }

    public void train(List<Image> images) {
        for (Image image : images) {
            List<double[][]> imList = new ArrayList<>();
            imList.add(multiply(image.getData(), 1.0/scaleFactor));

            double[] out = layers.getFirst().getOutput(imList);
            double[] dldO = getErrors(out, image.getLabel());

            layers.getLast().backPropagate(dldO);
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NEURAL_NETWORK\n");
        sb.append(ID).append("\n");
        sb.append(scaleFactor).append("\n");
        for (Layer layer : layers) {
            sb.append(layer.toString());
        }
        return sb.toString();
    }
}
