using System.Linq;

namespace Minami.NeuralNetwork
{
    public class ReluLayer : Layer
    {
        public bool[] Mask { get; set; }

        public ReluLayer() { }

        public double[] Forward(double[] input)
        {
            Mask = input.Select(x => x <= 0).ToArray();
            return input.Select((x, i) => Mask[i] ? 0 : x).ToArray();
        }

        public double[] Backword(double[] dout)
        {
            return dout.Select((x, i) => Mask[i] ? 0 : x).ToArray();
        }
    }
}
