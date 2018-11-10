using System;

namespace Minami.NeuralNetwork
{
    [Serializable]
    public class Bias
    {
        public double[] Hidden { get; set; }
        public double[] Output { get; set; }

        public Bias(UnitSize unitSize)
        {
            Hidden = new double[unitSize.Hidden];
            Output = new double[unitSize.Output];
        }
    }
}
