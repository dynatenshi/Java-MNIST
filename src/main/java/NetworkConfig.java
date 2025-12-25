import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NetworkConfig {
    public static final int SEED = 123;
    public static final double LEARNING_RATE = 0.001;
    public static final int IMG_SIZE = 28;
    public static final int CHANNELS = 1;

    public static MultiLayerConfiguration getConf() {
        return new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .updater(new Adam(LEARNING_RATE))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5).nIn(CHANNELS).nOut(20).stride(1, 1).activation(Activation.RELU).build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
                .layer(new ConvolutionLayer.Builder(5, 5).nOut(50).stride(1, 1).activation(Activation.RELU).build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
                .layer(new DenseLayer.Builder().nOut(500).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutionalFlat(IMG_SIZE, IMG_SIZE, CHANNELS))
                .build();
    }
}