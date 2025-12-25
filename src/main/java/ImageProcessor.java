import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ImageProcessor {
    private static final double WHITE_THRESHOLD = 127.0;
    private static final double BINARY_THRESHOLD = 110.0;
    private static final double MAX_COLOR = 255.0;

    public static INDArray prepareImage(INDArray image) {
        if (image == null) return null;

        if (image.meanNumber().doubleValue() > WHITE_THRESHOLD) {
            image = Nd4j.ones(image.shape()).mul(MAX_COLOR).sub(image);
        }
        for (int i = 0; i < image.length(); i++) {
            image.putScalar(i, image.getDouble(i) > BINARY_THRESHOLD ? MAX_COLOR : 0.0);
        }
        return centerImage(image).divi(MAX_COLOR);
    }

    private static INDArray centerImage(INDArray image) {
        int size = NetworkConfig.IMG_SIZE;
        INDArray core = image.get(NDArrayIndex.point(0), NDArrayIndex.point(0));
        int minR = size, maxR = 0, minC = size, maxC = 0;
        boolean found = false;

        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) {
                if (core.getDouble(r, c) > 0) {
                    minR = Math.min(minR, r); maxR = Math.max(maxR, r);
                    minC = Math.min(minC, c); maxC = Math.max(maxC, c);
                    found = true;
                }
            }
        }

        if (!found) return image;

        INDArray centered = Nd4j.zeros(1, 1, size, size);
        int h = maxR - minR + 1, w = maxC - minC + 1;
        int sR = (size - h) / 2, sC = (size - w) / 2;

        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                centered.putScalar(new int[]{0, 0, sR + r, sC + c}, core.getDouble(minR + r, minC + c));
            }
        }
        return centered;
    }
}