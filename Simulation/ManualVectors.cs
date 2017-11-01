using MathNet.Numerics.Distributions;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Simulation
{
    class ManualVectors
    {  //Serial/Parallel
        private const bool serial = false;
        // SDE Params
        private const double lambda = 2.5;
        private const double mu = 1;
        private const double T = 1;
        private static double[,] eta = { { 2 / Math.Sqrt(10), 1 / Math.Sqrt(10) }, { 1 / Math.Sqrt(10), 2 / Math.Sqrt(10) } };
        private const int n2steps = 12;
        private static int[] Narr = new int[n2steps];
        private static int Nmax;
        private const int M = 10000;


        static void Main(string[] args)
        {
            //MC Params
            for (int i = 0; i < n2steps; i++) { Narr[i] = (int)Math.Pow(2, i); }
            Nmax = Narr[Narr.Length - 1];

            //Run Calcs and only save end values
            double[] Xs = new double[n2steps];

            var watch = System.Diagnostics.Stopwatch.StartNew();

            if (serial)
            {
                Console.WriteLine("Running in Serial");
                for (int m = 0; m < M; m++)
                {
                    double[] MC_path = RunMC();
                    Xs = Xs.Select((x, index) => x + MC_path[index]).ToArray();
                }
            }
            else
            {
                Console.WriteLine("Running in Parallel");

                //object mylock = new object();

                //Parallel.For(0, M,
                //    () => new double[n2steps],
                //(m, loopstate, vec) =>
                //{
                //    double[] MC_path = RunMC();
                //    vec =  vec.Select((x,index) => x+ MC_path[index]).ToArray();
                //    return vec;
                //},
                //(vec) =>
                //{
                //    lock (mylock) { Xs = Xs.Select((x, index) => x + vec[index]).ToArray(); }
                //}
                //);

                ConcurrentBag<double[]> paths = new ConcurrentBag<double[]>();

                Parallel.For(0, M, m =>
                  {
                      double[] MCpath = RunMC();
                      paths.Add(MCpath);
                  }
                 );

                double[][] mc_paths = paths.ToArray();
                for(int i =0; i<M; i++)
                {
                    Xs = Xs.Select((x, index) => x + mc_paths[i][index]).ToArray();
                }                

            }
            watch.Stop();
            TimeSpan elapsedMs = watch.Elapsed;
            Xs = Xs.Select(x => Math.Sqrt(x / M)).ToArray();
            foreach (var item in Xs)
                Console.WriteLine(item.ToString("#.######"));
            Console.WriteLine("Elapsed time = " + elapsedMs.ToString("mm\\:ss\\.ff"));
            Console.ReadKey();

        }

        private static double[] RunMC()
        {
            double[] Xs = new double[n2steps];
            double[,] Xn = new double[2, n2steps];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < n2steps; j++)
                    Xn[i, j] = 1;

            //Generate Random Number Matrix
            double[] randn1 = new double[Nmax];
            double[] randn2 = new double[Nmax];
            Normal.Samples(randn1, 0, 1); Normal.Samples(randn2, 0, 1);

            for (int n = 0; n < Nmax; n++)
            {
                for (int i = 0; i < n2steps; i++)
                {
                    if (n % Narr[Narr.Length - 1 - i] == 0)
                    {
                        double[] X = { Xn[0, i], Xn[1, i] };
                        double[] z = { randn1[n], randn2[n] };
                        double h = T / Narr[i];
                        double tamedCoeff = 1 / (1 + Math.Pow(n, -1 / 2) * L2Norm(X));
                        double[] etaZ = MatrixVecMult(eta, z);
                        double l2nX = L2Norm(X);
                        
                        for (int j = 0; j < X.Length; j++)
                        {
                            Xn[j, i] += X[j] * (lambda * (mu - l2nX)) * h * tamedCoeff + etaZ[j] * tamedCoeff * Math.Pow(l2nX, 3 / 2) * Math.Sqrt(h);
                        }
                    }
                }
            }

            double[] XnLastCol = { Xn[0, Xn.GetLength(1) - 1], Xn[1, Xn.GetLength(1) - 1] };

            for (int i = 0; i < n2steps; i++)
            {
                Xs[i] = Math.Pow(L2Norm(XnLastCol.Select((x, index) => x - Xn[index, i]).ToArray()), 2);
            }

            return Xs;
        }

        private static double L2Norm(double[] arr)
        {
            double sum = 0;
            for (int i = 0; i < arr.Length; i++)
                sum += arr[i] * arr[i];
            return Math.Sqrt(sum);
        }

        private static double[] MatrixVecMult(double[,] Matrix, double[] Vec)
        {
            double[] result = new double[Vec.Length];

            for (int r = 0; r < Matrix.GetLength(0); r++)
            {
                for (int c = 0; c < Matrix.GetLength(1); c++)
                {
                    double tmp = 0;
                    for (int v = 0; v < Vec.Length; v++)
                    {
                        tmp += Matrix[r, v] * Vec[v];
                    }
                    result[r] = tmp;
                }
            }

            return result;
        }
    }
}


