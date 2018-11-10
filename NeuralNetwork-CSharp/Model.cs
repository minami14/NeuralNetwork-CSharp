using System;

namespace Minami.NeuralNetwork
{
    [Serializable]
    public class Model
    {
        public UnitSize unitSize { get; set; }
        public Bias bias { get; set; }
        public Weight weight { get; set; }

        public Model(UnitSize unitSize, Bias bias, Weight weight)
        {
            this.unitSize = unitSize;
            this.bias = bias;
            this.weight = weight;
        }
    }
}
