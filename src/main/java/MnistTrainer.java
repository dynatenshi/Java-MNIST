import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import java.io.File;

public class MnistTrainer {
    public static final String MODEL_PATH = "trained_mnist_model.zip";
    private static final int BATCH_SIZE = 64;
    private static final int EPOCHS = 5;

    public static void main(String[] args) {
        try {
            var trainData = new MnistDataSetIterator(BATCH_SIZE, true, NetworkConfig.SEED);
            MultiLayerNetwork model = new MultiLayerNetwork(NetworkConfig.getConf());
            model.init();
            model.setListeners(new ScoreIterationListener(100));

            for (int i = 0; i < EPOCHS; i++) {
                model.fit(trainData);
                trainData.reset();
            }
            model.save(new File(MODEL_PATH), true);
            System.out.println("Обучение завершено. Модель сохранена.");
        } catch (Exception e) {
            System.err.println("Ошибка при обучении: " + e.getMessage());
        }
    }
}