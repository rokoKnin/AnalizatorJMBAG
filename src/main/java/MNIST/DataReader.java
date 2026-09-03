package MNIST;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {

    private final int rows = 28;
    private final int cols = 28;

    public List<Image> readData(String path) {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            return br.lines()
                    .parallel()
                    .map(line -> {
                        String[] imageData = line.split(",");
                        double[][] data = new double[rows][cols];
                        int label = Integer.parseInt(imageData[0]);
                        int idx = 1;
                        for (int row = 0; row < rows; row++) {
                            for (int column = 0; column < cols; column++) {
                                data[row][column] = Double.parseDouble(imageData[idx++]);
                            }
                        }
                        return new Image(data, label);
                    })
                    .collect(java.util.stream.Collectors.toList());
        } catch (Exception e) {
            System.out.println("Error while reading file: " + e.getMessage());
            return new ArrayList<>();
        }
    }
}
