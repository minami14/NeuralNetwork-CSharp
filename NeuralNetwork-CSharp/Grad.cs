namespace Minami.NeuralNetwork
{
    public class Grad
    {
        public double[][] WeightInToHidden { get; set; }
        public double[][] WeightHiddenToOut { get; set; }
        public double BiasHidden { get; set; }
        public double BiasOut { get; set; }

        public Grad() { }
    }
}
