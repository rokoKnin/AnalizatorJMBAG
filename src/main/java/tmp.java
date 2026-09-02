import java.text.DecimalFormat;
import java.util.Random;

public class tmp {
    public static void main(){
        Random rand = new Random();
        DecimalFormat df = new DecimalFormat("0000000000 ");

        for (int i = 0; i < 100; i++) {
            System.out.printf("%s\n", df.format(rand.nextLong(10000, 999999999)));
        }
    }
}
