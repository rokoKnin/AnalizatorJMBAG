package Layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{
    private final long SEED;
    private final double LEAK = 0.01;

    private double[][] weights;
    private double[] biases;
    private int _inLength;
    private int _outLength;
    private double learningRate;

    private double[] lastZ;
    private double[] lastX;

    public FullyConnectedLayer(int _inLength, int _outLength, long SEED, double learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this.learningRate = learningRate;


        weights = new double[_inLength][_outLength];
        biases = new double[_outLength];
        setRandomWeights();

    }

    public FullyConnectedLayer(int _inLength, int _outLength, long SEED, double learningRate, double[][] weights, double[] biases) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.learningRate = learningRate;
        this.weights = weights;
        this.biases = biases;
        this.SEED = SEED;
    }

    public double[] fullyConnectedForwardPass(double[] input) {
        lastX = input;

        double[] z = new double[_outLength];
        System.arraycopy(biases, 0, z, 0, _outLength);
        double[] out = new double[_outLength];

        java.util.stream.IntStream.range(0, _outLength).parallel().forEach(j -> {
            double sum = biases[j];
            for (int i = 0; i < _inLength; i++) {
                sum += input[i] * weights[i][j];
            }
            z[j] = sum;
            out[j] = reLu(sum);
        });

        lastZ = z;

        for (int j = 0; j < _outLength; j++) {
            out[j] = reLu(z[j]);
        }

        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);

        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        if(_nextLayer != null) {
            return _nextLayer.getOutput(forwardPass);
        } else  {
            return forwardPass;
        }
    }

    @Override
    public void backPropagate(List<double[][]> dldO) {
        double[] vector = matrixToVector(dldO);

        backPropagate(vector);
    }

    @Override
    public void backPropagate(double[] dldO) {
        double[] dldx = new double[_inLength];

        double dOdz;
        double dzdw;
        double dldw;
        double dzdx;

        for (int j = 0; j < _outLength; j++) {
            dOdz = derrivativeReLu(lastZ[j]);
            double dldb = dldO[j] * dOdz;
            biases[j] -= learningRate * dldb;
        }

        for (int i = 0; i < _inLength; i++) {

            double dLdx_sum = 0;

            for (int j = 0; j < _outLength; j++) {

                dOdz = derrivativeReLu(lastZ[j]);
                dzdw = lastX[i];
                dzdx = weights[i][j];

                dldw = dldO[j]*dOdz*dzdw;

                weights[i][j] -= learningRate*dldw;

                dLdx_sum += dldO[j]*dOdz*dzdx;
            }

            dldx[i] = dLdx_sum;
        }
        if(_prevLayer != null) {
            _prevLayer.backPropagate(dldx);
        }
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outLength;
    }

    public void setRandomWeights() {
        Random rand = new Random(SEED);

        double scale = Math.sqrt(2.0 / _inLength);

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                weights[i][j] = rand.nextGaussian() * scale;
            }
        }

        for (int j = 0; j < _outLength; j++) {
            biases[j] = 0.01;
        }
    }

    public double reLu(double input) {
        return input < 0 ? LEAK : input;
    }

    public double derrivativeReLu(double input) {
        return input < 0 ? LEAK : 1;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("FULLY_CONNECTED\n");
        sb.append(_inLength).append(",").append(_outLength).append(",").append(learningRate).append(",").append(SEED).append("\n");

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                sb.append(weights[i][j]).append(",");
            }
            sb.append("\n");
        }
        for (int i = 0; i < _outLength; i++) {
            sb.append(biases[i]).append(",");
        }
        sb.append("\n");
        return sb.toString();
    }
}
