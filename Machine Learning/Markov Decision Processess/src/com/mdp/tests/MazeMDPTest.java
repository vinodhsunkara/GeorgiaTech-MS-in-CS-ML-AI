package com.mdp.tests;


import rl.DecayingEpsilonGreedyStrategy;
import rl.EpsilonGreedyStrategy;
import rl.MazeMarkovDecisionProcess;
import rl.MazeMarkovDecisionProcessVisualization;
import rl.Policy;
import rl.PolicyIteration;
import rl.QLambda;
import rl.SarsaLambda;
import rl.ValueIteration;
import shared.ConvergenceTrainer;
import shared.FixedIterationTrainer;
import shared.ThresholdTrainer;

/**
 * Tests out the maze markov decision process classes
 * @author guillory
 * @version 1.0
 */
public class MazeMDPTest {
    /**
     * Tests out things
     * @param args ignored
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        MazeMarkovDecisionProcess maze = MazeMarkovDecisionProcess.load("lib/testmaze_big.txt");
        System.out.println(maze);
        
        ValueIteration vi = new ValueIteration(.95, maze);
        ThresholdTrainer tt = new ThresholdTrainer(vi);
        ConvergenceTrainer ct = new ConvergenceTrainer(vi, .9, 100);
        long startTime = System.currentTimeMillis();
        tt.train();
        Policy p = vi.getPolicy();
        long finishTime = System.currentTimeMillis();
        System.out.println("Value iteration learned : " + p);
        System.out.println("in " + tt.getIterations() + " iterations");
        System.out.println("and " + (finishTime - startTime) + " ms");
        MazeMarkovDecisionProcessVisualization mazeVis =
            new MazeMarkovDecisionProcessVisualization(maze);
        System.out.println(mazeVis.toString(p));

        PolicyIteration pi = new PolicyIteration(.95, maze);
        tt = new ThresholdTrainer(pi);
         ct = new ConvergenceTrainer(pi, 3, 100);
        startTime = System.currentTimeMillis();
        ct.train();
        p = pi.getPolicy();
        finishTime = System.currentTimeMillis();
        System.out.println("Policy iteration learned : " + p);
        System.out.println("in " + ct.getIterations() + " iterations");
        System.out.println("and " + (finishTime - startTime) + " ms");
        System.out.println(mazeVis.toString(p));
        
        int iterations = 100;
       QLambda ql = new QLambda(.5, .95, .2, 1, new DecayingEpsilonGreedyStrategy(.3, 1.2), maze);
       // QLambda ql = new QLambda(.5, .95, .2, 1, new EpsilonGreedyStrategy(.3), maze);
      //  QLambda ql = new QLambda(.5, .95, .2, 1, new EpsilonGreedyStrategy(.3), maze);

      // ct = new ConvergenceTrainer(ql, .9, 50000);
        FixedIterationTrainer fit = new FixedIterationTrainer(ql, iterations);
        startTime = System.currentTimeMillis();
        fit.train();
        p = ql.getPolicy();
        finishTime = System.currentTimeMillis();
        System.out.println("Q lambda learned : " + p);
        System.out.println("in " +iterations + " iterations");
        System.out.println("and " + (finishTime - startTime) + " ms");
        System.out.println("Acquiring " + ql.getTotalReward() + " reward");
        System.out.println(mazeVis.toString(p));
        
        /**
        SarsaLambda sl = new SarsaLambda(.5, .95, .2, 1, new EpsilonGreedyStrategy(.3), maze);
        fit = new FixedIterationTrainer(sl, iterations);
        startTime = System.currentTimeMillis();
        fit.train();
        p = sl.getPolicy();
        finishTime = System.currentTimeMillis();
        System.out.println("Sarsa lambda learned : " + p);
        System.out.println("in " + iterations + " iterations");
        System.out.println("and " + (finishTime - startTime) + " ms");
        System.out.println("Acquiring " + sl.getTotalReward() + " reward");
        System.out.println(mazeVis.toString(p));
**/
    }

}
