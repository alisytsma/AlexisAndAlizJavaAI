/*
 Copyright (c) Ron Coleman

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package javaai.ann.learn.meta.ga;

import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.genetic.genome.IntegerArrayGenome;
import java.util.Random;

import static javaai.util.Helper.asInt;

/**
 * This class calculates the fitness of an individual chromosome or phenotype.
 */
class XorObjective implements CalculateScore {

    public final static boolean DEBUGGING = false;
    public final static String TEAM = "Alexis and Ali";
    public final static int NUM_WEIGHTS = 8;
    public final static double RANGE_MAX = 10.0;
    public final static double RANGE_MIN = -10.0;
    protected static Random ran = null;
    public final static double[][] XOR_INPUTS = {{0,0},{0,1},{1,0},{1,1}};
    public final static double[][] XOR_IDEALS = {{0},{1},{1},{0}};


    static {
        long seed = System.nanoTime();
        if(DEBUGGING)
            seed = TEAM.hashCode();
        ran = new Random(seed);
    }


    public static void main(String args[]){
        double[] ws = new double[NUM_WEIGHTS];

        for(int k=0; k < ws.length; k++)
            ws[k] = getRandomWeight();

        System.out.println(TEAM);
        for(int k=0; k < ws.length - 2; k++)
            System.out.printf("%5s %3.5f \n", ("w" + (k + 1)) + ": ", ws[k]);
        System.out.printf("%5s %3.5f \n", "b1: ", ws[6]);
        System.out.printf("%5s %3.5f \n", "b2: ", ws[7]);

        XorObjective objective = new XorObjective();

        System.out.printf("%3s %6s %6s %6s %6s \n", "#", "x1", "x2","t1","y1");

        for(int i = 0; i < XOR_INPUTS.length; i++){
            double y1 = objective.feedforward(XOR_INPUTS[i][0], XOR_INPUTS[i][1], ws);
            System.out.printf("%3d %1.4f %1.4f %1.4f %1.4f \n", i, XOR_INPUTS[i][0], XOR_INPUTS[i][1], XOR_IDEALS[i][0], y1);
        }

        double fitness = objective.getFitness(ws);
        System.out.println("Fitness: " + fitness);

    }
    /**
     * Calculates the fitness.
     * @param phenotype Individual
     * @return Objective
     */
    @Override
    public double calculateScore(MLMethod phenotype) {
        IntegerArrayGenome genome = (IntegerArrayGenome) phenotype;

        int x = asInt(genome);

        double y = f(x);

        return y;
    }

    /**
     * Specifies the objective
     * @return True to minimize, false to maximize.
     */
    @Override
    public boolean shouldMinimize() {
        return true;
    }

    /**
     * Specifies the threading approach.
     * @return True to use single thread, false for multiple threads
     */
    @Override
    public boolean requireSingleThreaded() {
        return true;
    }

    /**
     * Objective function
     * @param x Domain parameter.
     * @return y
     */
    protected int f(int x) {
        return (x - 3)*(x - 3);
    }

    /**
     * Returns a random weight.
     * @return double
     */
    public static double getRandomWeight() {
        double wt = ran.nextDouble()*(RANGE_MAX-RANGE_MIN)+RANGE_MIN;
        return wt;
    }

    /**
     * Returns the activated feedforward calculation.
     * @param x double
     * @return double
     */
    protected double sigmoid(double x){
        return 1.0 / (1 + Math.exp(-(x)));
    }

    /**
     * Calculates the ANN output.
     * @param x1 double
     * @param x2 double
     * @param ws double[]
     * @return double
     */
    protected double feedforward(double x1, double x2, double[] ws){

        double w1 = ws[0];
        double w2 = ws[1];
        double w3 = ws[2];
        double w4 = ws[3];
        double w5 = ws[4];
        double w6 = ws[5];
        double b1 = ws[6];
        double b2 = ws[7];

        double h1 = sigmoid(w1*x1 + w3*x2 + b1*1);
        double h2 = sigmoid(w2*x1 + w4*x2 + b1*1);
        double y1 = sigmoid(h1*w5 + h2*w6 + b2*1);

        return y1;

    }

    /**
     * Calculates the fitness.
     * @param ws double[]
     * @return double
     */
    public double getFitness(double[] ws) {
        double sumSqrErr = 0;
        double sqrErr = 0;
        for(int i = 0; i < XOR_INPUTS.length; i++){
            double y1 = feedforward(XOR_INPUTS[i][0], XOR_INPUTS[i][1], ws);
            sqrErr = (y1 - XOR_IDEALS[i][0]) * (y1 - XOR_IDEALS[i][0]);
        }

        sumSqrErr += sqrErr;
        double rmse = Math.sqrt(sumSqrErr / XOR_INPUTS.length);
        return rmse;
    }


}
