package Utils;

public class MatrixUtils {
    public static double[][] add(double[][] matrix1, double[][] matrix2){
        double[][] result = new double[matrix1.length][matrix1[0].length];
        for(int i=0;i<matrix1.length;i++){
            for(int j=0;j<matrix1[0].length;j++){
                result[i][j] = matrix1[i][j]+matrix2[i][j];
            }
        }
        return result;
    }
    public static double[] add(double[] matrix1, double[] matrix2){
        double[] result = new double[matrix1.length];
        for(int i=0;i<matrix1.length;i++){
            result[i] = matrix1[i]+matrix2[i];
        }
        return result;
    };

    public static double[][] multiply(double[][] matrix1, double[][] matrix2){
        double[][] result = new double[matrix1.length][matrix2[0].length];
        for(int i=0;i<matrix1.length;i++){
            for(int j=0;j<matrix2[0].length;j++){
                result[i][j] = matrix1[i][j]*matrix2[i][j];
            }
        }
        return result;
    }

    public static double[] multiply(double[] matrix1, double[] matrix2){
        double[] result = new double[matrix1.length];
        for(int i=0;i<matrix1.length;i++){
            result[i] = matrix1[i]*matrix2[i];
        }
        return result;
    }

    public static double[][] multiply(double[][] matrix, double konst) {
        double[][] result = new double[matrix.length][matrix[0].length];
        for(int i=0;i<matrix.length;i++){
            for(int j=0;j<matrix[0].length;j++){
                result[i][j] = matrix[i][j]*konst;
            }
        }
        return result;
    }
    public static double[] multiply(double[] matrix, double konst) {
        double[] result = new double[matrix.length];
        for(int i=0;i<matrix.length;i++){
            result[i] = matrix[i]*konst;
        }
        return result;
    }
}
