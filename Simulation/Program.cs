using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Data.Text;

namespace Simulation
{
    class Program
    {
        // SDE Params
        private static double lambda = 2.5;
        private static double mu = 1;
        private static double T = 1;
        private static double[,] etaArr = { { 2 / Math.Sqrt(10), 1 / Math.Sqrt(10) }, { 1 / Math.Sqrt(10), 2 / Math.Sqrt(10) } };
        private static Matrix<double> eta = Matrix<double>.Build.DenseOfArray(etaArr);

        private static Vector<double> CalculateOneStep(Vector<double> X, Vector<double> z, int n, double h)
        {
            double tamedCoeff = 1 / (1 + Math.Pow(n, -1 / 2) * X.L2Norm());

            return tamedCoeff * X * (lambda * (mu - X.L2Norm())) * h + tamedCoeff * eta * Math.Pow(X.L2Norm(), 3 / 2) * z * Math.Sqrt(h);
        }

        static void Main(string[] args)
        {
            //MC Params
            int n2steps = 10;
            int[] Narr = new int[n2steps];
            for (int i = 0; i < n2steps; i++) { Narr[i] = (int)Math.Pow(2, i); }
            int Nmax = Narr[Narr.Length - 1];
            int M = 1000;

            //Run Calcs and only save end values
            Matrix<double> Xs = Matrix<double>.Build.Dense(2, n2steps, 0); 
            Matrix<double> Xn = Matrix<double>.Build.Dense(2, n2steps, 1); 

            for (int m = 0; m < M; m++)
            {
                //Generate Random Number Matrix
                double[] randn1 = new double[Nmax];
                double[] randn2 = new double[Nmax];
                Normal.Samples(randn1, 0, 1); Normal.Samples(randn2, 0, 1);
                Matrix<double> randnMatrix = Matrix<double>.Build.DenseOfRowArrays(randn1, randn2);
                for(int n = 0; n < Nmax; n++)
                {
                    for(int i=0; i<n2steps; i++)
                    {
                        if (n % Narr[Narr.Length - 1 - i] == 0)
                        {
                            //Console.WriteLine("i = {0}, n = {1}", i, n);
                            Xn.SetColumn(i, Xn.Column(i) + CalculateOneStep(Xn.Column(i), randnMatrix.Column(n), n, T / Narr[i]));
                        }
                    }
                }
                Xs += Xn;
            }
            Xs *= 1 / M;
            //Console.ReadKey();
            DelimitedWriter.Write("../../../Xnp1.csv", Xn, ",");

        }
    }
}

