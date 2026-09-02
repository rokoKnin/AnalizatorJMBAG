package MNIST;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {

    private final int rows = 28;
    private final int cols = 28;

    public List<Image> readData(String path) {
        List<Image> images = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;

            while ((line = br.readLine()) != null) {
                String[] imageData = line.split(",");

                double[][] data = new double[rows][cols];

                int label = Integer.parseInt(imageData[0]);
                int i = 1;

                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < cols; column++) {
                        data[row][column] = (double) Integer.parseInt(imageData[i]);
                        i++;
                    }
                }

                images.add(new Image(data, label));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return images;
    }
}
