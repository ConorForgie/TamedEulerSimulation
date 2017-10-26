using System;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Data.Text;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace Simulation
{
    class Program
    {
        //Serial/Parallel
        private const bool serial = true;
        // SDE Params
        private const double lambda = 2.5;
        private const double mu = 1;
        private const double T = 1;
        private static double[,] etaArr = { { 2 / Math.Sqrt(10), 1 / Math.Sqrt(10) }, { 1 / Math.Sqrt(10), 2 / Math.Sqrt(10) } };
        private static Matrix<double> eta = Matrix<double>.Build.DenseOfArray(etaArr);
        private const int n2steps = 10;
        private static int[] Narr = new int[n2steps];
        private static int Nmax;
        private const int M = 1000;

        private static Vector<double> CalculateOneStep(Vector<double> X, Vector<double> z, int n, double h)
        {
            double tamedCoeff = 1 / (1 + Math.Pow(n, -1 / 2) * X.L2Norm());

            return tamedCoeff * X * (lambda * (mu - X.L2Norm())) * h + tamedCoeff * eta * Math.Pow(X.L2Norm(), 3 / 2) * z * Math.Sqrt(h);
        }

        private static Vector<double> RunMC()
        {
            Vector<double> Xs = Vector<double>.Build.Dense(n2steps, 0);
            Matrix<double> Xn = Matrix<double>.Build.Dense(2, n2steps, 1);

            //Generate Random Number Matrix
            double[] randn1 = new double[Nmax];
            double[] randn2 = new double[Nmax];
            Normal.Samples(randn1, 0, 1); Normal.Samples(randn2, 0, 1);
            Matrix<double> randnMatrix = Matrix<double>.Build.DenseOfRowArrays(randn1, randn2);
            for (int n = 0; n < Nmax; n++)
            {
                for (int i = 0; i < n2steps; i++)
                {
                    if (n % Narr[Narr.Length - 1 - i] == 0)
                    {
                        Xn.SetColumn(i, Xn.Column(i) + CalculateOneStep(Xn.Column(i), randnMatrix.Column(n), n, T / Narr[i]));
                    }
                }
            }

            for (int i = 0; i < n2steps; i++)
            {
                Xs[i] = Math.Pow((Xn.Column(Xn.ColumnCount - 1) - Xn.Column(i)).L2Norm(), 2);
            }

            return Xs;
        }

        static void Main(string[] args)
        {
            //MC Params
            for (int i = 0; i < n2steps; i++) { Narr[i] = (int)Math.Pow(2, i); }
            Nmax = Narr[Narr.Length - 1];

            //Run Calcs and only save end values
            Vector<double> Xs = Vector<double>.Build.Dense(n2steps, 0);

            if (serial)
            {
                for (int m = 0; m < M; m++)
                    Xs += RunMC();
            }
            else
            {
                object mylock = new object();

                Parallel.For(0, M,
                    () => Vector<double>.Build.Dense(n2steps, 0),
                (m, loopstate, vec) =>
                {
                    vec = RunMC();
                    return vec;
                },
                (vec) =>
                {
                    lock (mylock) { Xs = Xs + vec; }
                }
                );
            }

            Xs = Xs / M;
            Xs.PointwiseSqrt();
            Console.Write(Xs);
            Console.ReadKey();
            //DelimitedWriter.Wirte("../../../Xnp1.csv", Xn, ",");

        }
    }
}

