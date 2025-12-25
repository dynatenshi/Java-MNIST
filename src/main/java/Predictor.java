import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Predictor {
    private static final String INPUT_PATH = "src/main/resources/test_images";

    public static void main(String[] args) {
        try {
            File modelFile = new File(MnistTrainer.MODEL_PATH);
            if (!modelFile.exists()) {
                System.out.println("Ошибка: Файл модели не найден. Сначала запустите MnistTrainer.");
                return;
            }

            MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);
            NativeImageLoader loader = new NativeImageLoader(NetworkConfig.IMG_SIZE, NetworkConfig.IMG_SIZE, NetworkConfig.CHANNELS);

            File input = new File(INPUT_PATH);
            List<File> files = getPngFiles(input);

            if (files.isEmpty()) {
                System.out.println("Внимание: Не найдено .png файлов для обработки по пути: " + input.getAbsolutePath());
                return;
            }

            System.out.printf("%-20s | %-10s | %-10s%n", "Файл", "Цифра", "Уверенность");
            System.out.println("----------------------------------------------------");

            for (File f : files) {
                try {
                    INDArray raw = loader.asMatrix(f);
                    INDArray processed = ImageProcessor.prepareImage(raw);
                    int digit = model.predict(processed)[0];
                    double conf = model.output(processed).maxNumber().doubleValue() * 100;
                    System.out.printf("%-20s | %-10d | %.2f%%%n", f.getName(), digit, conf);
                } catch (Exception e) {
                    System.err.println("Ошибка обработки файла " + f.getName() + ": " + e.getMessage());
                }
            }
        } catch (Exception e) {
            System.err.println("Критическая ошибка: " + e.getMessage());
        }
    }

    private static List<File> getPngFiles(File root) {
        List<File> list = new ArrayList<>();
        if (root.isDirectory()) {
            File[] content = root.listFiles();
            if (content != null) {
                for (File f : content) {
                    if (f.getName().toLowerCase().endsWith(".png")) list.add(f);
                }
            }
        } else if (root.isFile() && root.getName().toLowerCase().endsWith(".png")) {
            list.add(root);
        }
        return list;
    }
}