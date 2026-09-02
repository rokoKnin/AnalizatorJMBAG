package Network;

import Layers.Layer;
import MNIST.Image;

import java.util.ArrayList;
import java.util.List;

import static Utils.MatrixUtils.add;
import static Utils.MatrixUtils.multiply;

public class NeuralNetwork {
    List<Layer> layers;

    public NeuralNetwork(List<Layer> layers) {
        this.layers = layers;

        linkLayers();
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

        imList.add(image.getData());

        double[] out = layers.getFirst().getOutput(imList);

        return getMaxIndex(out);
    }

    public float test (List<Image> images) {
        float correct = 0;

        for(Image image : images){
            int guess = guessMNIST(image);

            if (guess == image.getLabel()) {
                correct ++;
            }
        }

        return correct / (float)images.size();
    }

    public void train(List<Image> images) {
        for (Image image : images) {
            List<double[][]> imList = new ArrayList<>();
            imList.add(image.getData());

            double[] out = layers.getFirst().getOutput(imList);
            double[] dldO = getErrors(out, image.getLabel());

            layers.getLast().backPropagate(dldO);
        }
    }
}
