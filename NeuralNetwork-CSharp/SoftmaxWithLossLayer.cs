using System;
using System.Linq;

namespace Minami.NeuralNetwork
{
    public class SoftmaxWithLossLayer
    {
        public double Loss { get; set; }
        public double[] Teacher { get; set; }
        public double[] output { get; set; }

        public SoftmaxWithLossLayer() { }

        public double Forward(double[] input, double[] teacher)
        {
            Teacher = teacher;
            output = Softmax(input);
            Loss = CrossEntropyError(output, teacher);
            return Loss;
        }

        public double[] Backword(double dout = 1)
        {
            return output.Select((x, i) => (x - Teacher[i]) / Teacher.Length).ToArray();
        }

        public double CrossEntropyError(double[] output, double[] teacher)
        {
            var delta = 1e-7;
            return -output.Select((x, i) => teacher[i] * Math.Log(x + delta)).Sum();
        }

        public double[] Softmax(double[] arr)
        {
            var max = arr.Max();
            var exp = new double[arr.Length];
            var expSum = 0.0;
            for (var i = 0; i < arr.Length; i++)
            {
                exp[i] = Math.Exp(arr[i] - max);
                expSum += exp[i];
            }
            var result = new double[arr.Length];
            for (var i = 0; i < arr.Length; i++)
            {
                result[i] = exp[i] / expSum;
            }
            return result;
        }
    }
}
