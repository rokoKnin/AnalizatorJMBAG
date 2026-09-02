package MNIST;

public class Image {
    private double[][] data;
    private int label;

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public int getLabel() {
        return label;
    }

    public double[][] getData() {
        return data;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(label).append("\n");

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                sb.append(data[i][j]).append(", ");
            }
            sb.append("\n");
        }

        return sb.toString();
    }
}
