using System;
using System.Linq;

namespace Minami.NeuralNetwork
{
    public class AffineLayer : Layer
    {
        public double[][] weight { get; set; }
        public double[] bias { get; set; }
        public double[] Input { get; set; }
        public double[][] DeltaWeight { get; set; }
        public double DeltaBias { get; set; }

        public AffineLayer(double[][] weight, double[] bias)
        {
            this.weight = weight;
            this.bias = bias;
        }

        public double[] Forward(double[] input)
        {
            Input = new double[input.Length];
            Array.Copy(input, Input, input.Length);
            return Dot(new double[1][] { input }, weight)[0].Select((x, i) => x - bias[i]).ToArray();
        }

        public double[] Backword(double[] dout)
        {
            DeltaWeight = Dot(Transverse(new double[1][] { Input }), new double[1][] { dout });
            DeltaBias = dout.Sum() / dout.Length;
            return Dot(new double[1][] { dout }, Transverse(weight))[0];
        }

        public double[][] Dot(double[][] input, double[][] weight)
        {
            var result = new double[input.Length][];
            for (var i = 0; i < input.Length; i++)
            {
                result[i] = new double[weight[0].Length];
                for (var j = 0; j < weight[0].Length; j++)
                {
                    for (var k = 0; k < weight.Length; k++)
                    {
                        result[i][j] += input[i][k] * weight[k][j];
                    }
                }
            }
            return result;
        }

        public double[][] Transverse(double[][] matrix)
        {
            var transverseMatrix = new double[matrix[0].Length][];
            for (var i = 0; i < matrix[0].Length; i++)
            {
                transverseMatrix[i] = new double[matrix.Length];
                for (var j = 0; j < matrix.Length; j++)
                {
                    transverseMatrix[i][j] = matrix[j][i];
                }
            }
            return transverseMatrix;
        }
    }
}
