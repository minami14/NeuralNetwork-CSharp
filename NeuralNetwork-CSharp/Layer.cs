namespace Minami.NeuralNetwork
{
    public interface Layer
    {
        double[] Forward(double[] input);
        double[] Backword(double[] dout);
    }
}
