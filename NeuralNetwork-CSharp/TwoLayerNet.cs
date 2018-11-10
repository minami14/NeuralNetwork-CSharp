using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace Minami.NeuralNetwork
{
    public class TwoLayerNet
    {
        public UnitSize unitSize { get; set; }
        public Weight weight { get; set; }
        public Bias bias { get; set; }
        public Layer[] Layers { get; set; }
        private SoftmaxWithLossLayer LastLayer = new SoftmaxWithLossLayer();

        public TwoLayerNet(UnitSize unitSize)
        {
            this.unitSize = unitSize;
            weight = new Weight(unitSize);
            bias = new Bias(unitSize);
            Initialize();
            Layers = new Layer[3];
            Layers[0] = new AffineLayer(weight.InToHidden, bias.Hidden);
            Layers[1] = new ReluLayer();
            Layers[2] = new AffineLayer(weight.HiddenToOut, bias.Output);
        }

        public TwoLayerNet(Model model)
        {
            unitSize = model.unitSize;
            weight = model.weight;
            bias = model.bias;
            Layers = new Layer[3];
            Layers[0] = new AffineLayer(weight.InToHidden, bias.Hidden);
            Layers[1] = new ReluLayer();
            Layers[2] = new AffineLayer(weight.HiddenToOut, bias.Output);
        }

        public void Initialize(double weightInit = 0.001)
        {
            var random = new Random();
            for (var i = 0; i < unitSize.Input; i++)
            {
                for (var j = 0; j < unitSize.Hidden; j++)
                {
                    weight.InToHidden[i][j] = random.NextDouble() * weightInit;
                }
            }

            for (var i = 0; i < unitSize.Hidden; i++)
            {
                for (var j = 0; j < unitSize.Output; j++)
                {
                    weight.HiddenToOut[i][j] = random.NextDouble() * weightInit;
                }
            }

            for (var i = 0; i < unitSize.Hidden; i++)
            {
                bias.Hidden[i] = 0;
            }

            for (var i = 0; i < unitSize.Output; i++)
            {
                bias.Output[i] = 0;
            }
        }

        public double[] Predict(double[] input)
        {
            var output = new double[input.Length];
            Array.Copy(input, output, input.Length);
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        public double Loss(double[] input, double[] teacher)
        {
            var output = Predict(input);
            return LastLayer.Forward(output, teacher);
        }

        public Grad Gradient(double[] input, double[] teacher)
        {
            Loss(input, teacher);
            var dout = LastLayer.Backword();
            var reversedLayers = new Layer[Layers.Length];
            Array.Copy(Layers, reversedLayers, Layers.Length);
            Array.Reverse(reversedLayers);
            foreach (var layer in reversedLayers)
            {
                dout = layer.Backword(dout);
            }
            var grad = new Grad();
            grad.WeightInToHidden = ((AffineLayer)Layers[0]).DeltaWeight;
            grad.WeightHiddenToOut = ((AffineLayer)Layers[2]).DeltaWeight;
            grad.BiasHidden = ((AffineLayer)Layers[0]).DeltaBias;
            grad.BiasOut = ((AffineLayer)Layers[2]).DeltaBias;
            return grad;
        }

        public void Train(double[] trainData, double[] teacher, double learningRate)
        {
            var grad = Gradient(trainData, teacher);
            for (var j = 0; j < unitSize.Input; j++)
            {
                for (var k = 0; k < unitSize.Hidden; k++)
                {
                    ((AffineLayer)Layers[0]).weight[j][k] -= learningRate * grad.WeightInToHidden[j][k];
                }
            }
            for (var j = 0; j < unitSize.Hidden; j++)
            {
                for (var k = 0; k < unitSize.Output; k++)
                {
                    ((AffineLayer)Layers[2]).weight[j][k] -= learningRate * grad.WeightHiddenToOut[j][k];
                }
            }
            for (var j = 0; j < unitSize.Hidden; j++)
            {
                ((AffineLayer)Layers[0]).bias[j] -= learningRate * grad.BiasHidden;
            }
            for (var j = 0; j < unitSize.Output; j++)
            {
                ((AffineLayer)Layers[2]).bias[j] -= learningRate * grad.BiasOut;
            }
        }

        public void SaveModelToBinaryFile(string filePath)
        {
            var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            var bf = new BinaryFormatter();
            var model = new Model(unitSize, bias, weight);
            bf.Serialize(fs, model);
            fs.Close();
        }

        public static TwoLayerNet LoadModelFromBinaryFile(string filePath)
        {
            var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            var binaryFormatter = new BinaryFormatter();
            Model obj;
            try
            {
                obj = (Model)binaryFormatter.Deserialize(fileStream);
            }
            catch (System.Runtime.Serialization.SerializationException e)
            {
                fileStream.Close();
                throw e;
            }
            fileStream.Close();
            return new TwoLayerNet(obj);
        }
    }
}
