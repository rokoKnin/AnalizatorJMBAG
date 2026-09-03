package MNIST;
import Image.Image;
import Network.NeuralNetwork;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class ExecutorServiceTester {

    public static float runParallelTest(NeuralNetwork baseNetwork, List<Image> testImages, int numThreads) {
        // 1. Serialize the base network so it can be cloned[cite: 9]
        String serializedModel = baseNetwork.toString();

        // 2. Use ThreadLocal so each thread in the pool initializes its own isolated NeuralNetwork[cite: 9]
        ThreadLocal<NeuralNetwork> threadLocalModel = ThreadLocal.withInitial(() ->
                {
                    try {
                        return NeuralNetwork.fromString(serializedModel);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
        );

        // 3. Create the ExecutorService with a fixed number of worker threads
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Callable<Integer>> tasks = new ArrayList<>();

        // 4. Create a lightweight task for every single image
        for (Image img : testImages) {
            tasks.add(() -> {
                // Retrieves the isolated NeuralNetwork specific to the current executing thread
                NeuralNetwork localNN = threadLocalModel.get();

                // Return 1 if the guess matches the label, 0 if incorrect[cite: 6, 9]
                if (localNN.guess(img) == img.getLabel()) {
                    return 1;
                }
                return 0;
            });
        }

        int totalCorrect = 0;

        try {
            // 5. invokeAll submits all tasks and blocks until every task has finished
            List<Future<Integer>> results = executor.invokeAll(tasks);

            // 6. Tally up the correct predictions
            for (Future<Integer> result : results) {
                totalCorrect += result.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            Thread.currentThread().interrupt();
            e.printStackTrace();
        } finally {
            // 7. Always shut down the executor to prevent resource leaks
            executor.shutdown();
            threadLocalModel.remove();
        }

        return (float) totalCorrect / testImages.size();
    }
}
