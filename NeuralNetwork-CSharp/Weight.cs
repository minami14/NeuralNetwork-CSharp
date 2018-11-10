using System;
using System.Linq;

namespace Minami.NeuralNetwork
{
    [Serializable]
    public class Weight
    {
        public double[][] InToHidden { get; set; }
        public double[][] HiddenToOut { get; set; }

        public Weight(UnitSize unitSize)
        {
            InToHidden = new double[unitSize.Input][].Select(x => new double[unitSize.Hidden]).ToArray();
            HiddenToOut = new double[unitSize.Hidden][].Select(x => new double[unitSize.Output]).ToArray();
        }
    }
}
