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
            System.out.printf("%5s %3.5f \n", ("ws" + (k + 1)) + ": ", ws[k]);
        System.out.printf("%5s %3.5f \n", "b1: ", ws[6]);
        System.out.printf("%5s %3.5f \n", "b2: ", ws[7]);


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

    public static double getRandomWeight() {
        double wt = ran.nextDouble()*(RANGE_MAX-RANGE_MIN)+RANGE_MIN;
        return wt;
    }

}
