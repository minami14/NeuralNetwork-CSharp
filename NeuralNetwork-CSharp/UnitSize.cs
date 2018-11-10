using System;

namespace Minami.NeuralNetwork
{
    [Serializable]
    public class UnitSize
    {
        public int Input { get; set; }
        public int Hidden { get; set; }
        public int Output { get; set; }

        public UnitSize(int input, int hidden, int output)
        {
            Input = input;
            Hidden = hidden;
            Output = output;
        }
    }
}
