package Network;

import Layers.ConvolutionLayer;
import Layers.FullyConnectedLayer;
import Layers.Layer;
import Layers.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
    private NeuralNetwork nn;
    private int _inputRows;
    private int _inputColumns;
    private double _scaleFactor;
    List<Layer> layers;

    public NetworkBuilder(int _inputRows, int _inputColumns, double _scaleFactor) {
        this._inputRows = _inputRows;
        this._inputColumns = _inputColumns;
        this._scaleFactor = _scaleFactor;

        layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long SEED) {
        if (layers.isEmpty()) {
            layers.add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows, _inputColumns, SEED, numFilters, learningRate));
        } else {
            Layer prev = layers.getLast();
            layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED, numFilters, learningRate));
        }
    }

    public void addMaxPoolLayer(int windowSize, int stepSize) {
        if (layers.isEmpty()) {
            layers.add(new MaxPoolLayer(stepSize, windowSize, 1, _inputRows, _inputColumns));
        } else {
            Layer prev = layers.getLast();
            layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    public void addFullyConnectedLayer(int outLength, double learningRate, long SEED) {
        if (layers.isEmpty()) {
            layers.add(new FullyConnectedLayer(_inputRows*_inputColumns, outLength, SEED, learningRate));
        } else {
            Layer prev = layers.getLast();
            layers.add(new FullyConnectedLayer(prev.getOutputElements(), outLength, SEED, learningRate));
        }
    }

    public NeuralNetwork build() {
        nn = new NeuralNetwork(layers, _scaleFactor);
        return nn;
    }
}
