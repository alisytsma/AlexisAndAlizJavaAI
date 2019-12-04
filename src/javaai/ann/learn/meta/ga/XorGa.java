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
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.BasicPopulation;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.species.BasicSpecies;
import org.encog.ml.ea.species.Species;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.ml.genetic.crossover.Splice;
import org.encog.ml.genetic.genome.DoubleArrayGenome;
import org.encog.ml.genetic.genome.DoubleArrayGenomeFactory;
import org.encog.ml.genetic.genome.IntegerArrayGenome;
import org.encog.ml.genetic.genome.IntegerArrayGenomeFactory;

import java.util.Random;

import static javaai.util.Helper.asString;

/**
 * This class uses a genetic algorithm for unsupervised learning to solve y = (x-3)^2.
 */
public class XorGa {

    /** Stopping criteria as difference between best solution and last best one */
    public final double TOLERANCE = 0.01;

    /** Convergence criteria: number of times best solution stays best */
    public final static int MAX_SAME_COUNT = 100;

    /** Population has this many individuals. */
    public final static int POPULATION_SIZE = 10000;

    /** Chromosome size (ie, number of genes): domain is [0, (2^n-1)]. */
    public final static int GENOME_SIZE = 8;

    /** Mutation rate */
    public final static double MUTATION_RATE = 0.01;

    /** Same count for test of convergence */
    protected int sameCount = 0;

    /** Last y value of training iteration */
    protected double yLast;

    /**
     * Runs the program.
     * @param args Command line arguments not used.
     */
    public static void main(String[] args) {
        XorGa ga = new XorGa();

        DoubleArrayGenome best = ga.solve();
    }

    /**
     * Solves the objective.
     * @return Best individual
     */
    public DoubleArrayGenome solve() {
        // Initialize a population
        Population population = initPop();
        // output("before", population);

        // Get the fitness measure
        CalculateScore objective = new XorObjective();

        // Create the evolutionary training algorithm
        TrainEA ga = new TrainEA(population, objective);

        // Set the mutation rate: 2nd operation tends to give better results.
        // ga.addOperation(MUTATION_RATE, new MutateShuffle());
        ga.addOperation(MUTATION_RATE, new MutateDoubleArrayGenome(0.001));

        // Set up to splice along the middle of the genome
        ga.addOperation(0.9, new Splice(GENOME_SIZE /2));

        // Do the learning algorithm
        train(ga);

        // Return the best individual
        DoubleArrayGenome best = (DoubleArrayGenome)ga.getBestGenome();
        population = ga.getPopulation();

        // output("after", population);
        XorObjective xorObjective = new XorObjective();
        System.out.printf("%6s %6s %6s %6s \n", "x1", "x2","t1","y1");

        for(int i = 0; i < xorObjective.XOR_INPUTS.length; i++){
            double y1 = xorObjective.feedforward(xorObjective.XOR_INPUTS[i][0], xorObjective.XOR_INPUTS[i][1], best.getData());
            System.out.printf("%1.4f %1.4f %1.4f %1.4f \n", xorObjective.XOR_INPUTS[i][0], xorObjective.XOR_INPUTS[i][1], xorObjective.XOR_IDEALS[i][0], y1);
        }

        System.out.println("best =" + asString(best));

        System.out.println("fitness = " + xorObjective.getFitness(best.getData()));

        return best;
    }

    /**
     * Runs the learning algorithm.
     * @param ga
     */
    protected void train(TrainEA ga) {
        int iteration = 0;

        boolean converged = false;

        System.out.printf("%3s %5s %5s %7s", "#", "y1", "same", "best\n");
        // Loop until the best answer doesn't change for a while
        while(!converged) {
            // output("iteration = "+iteration, ga.getPopulation());

            ga.iteration();

            // Get the value of the best solution for predict(x)
            double y = ga.getError();

            DoubleArrayGenome best = (DoubleArrayGenome) ga.getBestGenome();

            System.out.printf("%3d %5.2f %5d %s\n",iteration, y, sameCount, asString(best));

            iteration++;

            converged = didConverge(y,  ga.getPopulation());
        }
    }

    /**
     * Tests whether GA has converged.
     * @param y Y value in y=predict(x)
     * @param pop Population of individuals
     * @return True if the GA has converge, otherwise false
     */
    public boolean didConverge(double y, Population pop) {
        if(sameCount >= MAX_SAME_COUNT)
            return true;

        if(Math.abs(yLast - y) < TOLERANCE) {
            sameCount++;
        }
        else
            sameCount = 0;

        yLast = y;

        return false;
    }

    /**
     * Initializes a population.
     * @return Population
     */
    protected Population initPop() {
        Population pop = new BasicPopulation(POPULATION_SIZE, null);

        BasicSpecies species = new BasicSpecies();

        species.setPopulation(pop);

        for(int k=0; k < POPULATION_SIZE; k++) {
            final DoubleArrayGenome genome = randomGenome(GENOME_SIZE);

            species.getMembers().add(genome);
        }

        pop.setGenomeFactory(new DoubleArrayGenomeFactory(GENOME_SIZE));
        pop.getSpecies().add(species);

        return pop;
    }

    /**
     * Gets a random individual
     * @param sz Number of genes
     * @return
     */
    public DoubleArrayGenome randomGenome(int sz) {
        DoubleArrayGenome genome = new DoubleArrayGenome(sz);

        final double[] organism = genome.getData();

        for(int k=0; k < organism.length; k++) {
            organism[k] = XorObjective.getRandomWeight();
        }

        return genome;
    }

    /**
     * Dumps the population.
     * @param title Title
     * @param pop Population
     */
    protected void output(final String title, final Population pop) {
        final Species species = pop.getSpecies().get(0);

        System.out.println("----- "+title);

        int n = 1;

        for (Genome genome : species.getMembers()) {
            DoubleArrayGenome individual = (DoubleArrayGenome) genome;

            System.out.printf("%3d | %s\n",n, asString(individual));

            n++;
        }
    }
}


