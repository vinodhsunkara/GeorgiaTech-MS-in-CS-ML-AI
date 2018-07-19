/**
 * 
 */
package com.mdp.tests;

import burlap.behavior.singleagent.learning.GoalBasedRF;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.vfa.common.ConcatenatedObjectFeatureVectorGenerator;
import burlap.behavior.singleagent.vfa.fourier.FourierBasis;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.visualizer.Visualizer;

/**
 * @author kmanda1
 * 
 */
public class MountainCarTest {

	/**
	 * A constant for the name of the x attribute
	 */
	public static final String ATTX = "xAtt";

	/**
	 * A constant for the name of the velocity attribute
	 */
	public static final String ATTV = "vAtt";

	/**
	 * A constant for the name of the coast action
	 */
	public static final String ACTIONCOAST = "coast";

	/**
	 * A constant for the name of the agent class
	 */
	public static final String CLASSAGENT = "agent";

	/**
	 * A constant for the name of the forward action
	 */
	public static final String ACTIONFORWARD = "forward";

	/**
	 * A constant for the name of the backwards action
	 */
	public static final String ACTIONBACKWARDS = "backwards";

	/**
	 * 
	 */
	public MountainCarTest() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {

		/**
		 * MountainCar mcGen = new MountainCar(); Domain domain =
		 * mcGen.generateDomain(); State s = mcGen.getCleanState(domain);
		 * 
		 * 
		 * Visualizer vis = MountainCarVisualizer.getVisualizer(mcGen);
		 * VisualExplorer exp = new VisualExplorer(domain, vis, s);
		 * 
		 * exp.addKeyAction("d", ACTIONFORWARD); exp.addKeyAction("s",
		 * ACTIONCOAST); exp.addKeyAction("a", ACTIONBACKWARDS);
		 * 
		 * exp.initGUI();
		 **/

		
		MountainCar mcGen = new MountainCar();
		Domain domain = mcGen.generateDomain();
		TerminalFunction tf = mcGen.new ClassicMCTF();
		RewardFunction rf = new GoalBasedRF(tf, 100);

		StateGenerator rStateGen = new MCRandomStateGenerator(domain);
		SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(
				domain);
		SARSData dataset = collector.collectNInstances(rStateGen, rf, 5000, 20,
				tf, null);

		ConcatenatedObjectFeatureVectorGenerator featureVectorGenerator = new ConcatenatedObjectFeatureVectorGenerator(
				true, MountainCar.CLASSAGENT);
		FourierBasis fb = new FourierBasis(featureVectorGenerator, 4);

		LSPI lspi = new LSPI(domain, rf, tf, 0.99, fb);
		lspi.setDataset(dataset);

		lspi.runPolicyIteration(30, 1e-6);

		GreedyQPolicy p = new GreedyQPolicy(lspi);

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vexp = new VisualActionObserver(domain, v);
		vexp.initGUI();
		((SADomain) domain).addActionObserverForAllAction(vexp);

		State s = mcGen.getCleanState(domain);
		for (int i = 0; i < 5; i++) {
			p.evaluateBehavior(s, rf, tf);
		}

		System.out.println("Finished.");
		
		
		
		/**
		GreedyQPolicy p = new GreedyQPolicy(lspi);

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vexp = new VisualActionObserver(domain, v);
		vexp.initGUI();
		((SADomain)domain).addActionObserverForAllAction(vexp);

		State s = mcGen.getCleanState(domain);
		for(int i = 0; i < 5; i++){
			p.evaluateBehavior(s, rf, tf);
		}

		System.out.println("Finished.");
		**/
	}
}
