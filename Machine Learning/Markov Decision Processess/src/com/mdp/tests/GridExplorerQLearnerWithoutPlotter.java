/**
 * 
 */
package com.mdp.tests;
import java.awt.Color;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.QValue;
import burlap.behavior.singleagent.ValueFunctionInitialization;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D.PolicyGlyphRenderStyle;
import burlap.behavior.singleagent.learning.GoalBasedRF;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.commonpolicies.EpsilonGreedy;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.deterministic.TFGoalCondition;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.oomdp.core.AbstractGroundedAction;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.UniformCostRF;

public class GridExplorerQLearnerWithoutPlotter extends OOMDPPlanner implements QComputablePlanner,
		LearningAgent {

	protected Map<StateHashTuple, List<QValue>> qValues;
	protected ValueFunctionInitialization qinit;
	protected double learningRate;
	protected Policy learningPolicy;
	
	protected LinkedList<EpisodeAnalysis> storedEpisodes = new LinkedList<EpisodeAnalysis>();
	protected int maxStoredEpisodes = 1;
	
	
	public GridExplorerQLearnerWithoutPlotter(Domain domain, RewardFunction rf, TerminalFunction tf, double gamma, 
		StateHashFactory hashingFactory, ValueFunctionInitialization qinit,
		double learningRate, double epsilon){

		this.plannerInit(domain, rf, tf, gamma, hashingFactory);
		this.qinit = qinit;
		this.learningRate = learningRate;
		this.qValues = new HashMap<StateHashTuple, List<QValue>>();
		this.learningPolicy = new EpsilonGreedy(this, epsilon);
	}
	
	@Override
	public EpisodeAnalysis runLearningEpisodeFrom(State initialState) {
		return this.runLearningEpisodeFrom(initialState, -1);
	}

	@Override
	public EpisodeAnalysis runLearningEpisodeFrom(State initialState,
			int maxSteps) {
		
		//initialize our episode analysis object with the given initial state
		EpisodeAnalysis ea = new EpisodeAnalysis(initialState);
		
		//behave until a terminal state or max steps is reached
		State curState = initialState;
		int steps = 0;
		while(!this.tf.isTerminal(curState) && (steps < maxSteps || maxSteps == -1)){
			
			//select an action
			AbstractGroundedAction a = this.learningPolicy.getAction(curState);
			
			//take the action and observe outcome
			State nextState = a.executeIn(curState);
			double r = this.rf.reward(curState, (GroundedAction)a, nextState);
			
			//record result
			ea.recordTransitionTo((GroundedAction)a, nextState, r);
			
			//update the old Q-value
			QValue oldQ = this.getQ(curState, a);
			oldQ.q = oldQ.q + this.learningRate * 
				(r + (this.gamma * this.maxQ(nextState) - oldQ.q));
			
			
			//move on to next state
			curState = nextState;
			steps++;
			
		}
		
		while(this.storedEpisodes.size() >= this.maxStoredEpisodes){
			this.storedEpisodes.poll();
		}
		this.storedEpisodes.offer(ea);
		
		return ea;
	}

	@Override
	public EpisodeAnalysis getLastLearningEpisode() {
		return this.storedEpisodes.getLast();
	}

	@Override
	public void setNumEpisodesToStore(int numEps) {
		this.maxStoredEpisodes = numEps;
		while(this.storedEpisodes.size() > this.maxStoredEpisodes){
			this.storedEpisodes.poll();
		}
	}

	@Override
	public List<EpisodeAnalysis> getAllStoredLearningEpisodes() {
		return this.storedEpisodes;
	}

	@Override
	public List<QValue> getQs(State s) {
		
		//first get hashed state
		StateHashTuple sh = this.hashingFactory.hashState(s);
		
		//check if we already have stored values
		List<QValue> qs = this.qValues.get(sh);
		
		//create and add initialized Q-values if we don't have them stored for this state
		if(qs == null){
			List<GroundedAction> actions = this.getAllGroundedActions(s);
			qs = new ArrayList<QValue>(actions.size());
			//create a Q-value for each action
			for(GroundedAction ga : actions){
				//add q with initialized value
				qs.add(new QValue(s, ga, this.qinit.qValue(s, ga)));
			}
			//store this for later
			this.qValues.put(sh, qs);
		}
		
		return qs;
	}

	@Override
	public QValue getQ(State s, AbstractGroundedAction a) {
		
		//first get all Q-values
		List<QValue> qs = this.getQs(s);
		
		//translate action parameters to source state for Q-values if needed
		a = a.translateParameters(s, qs.get(0).s);
		
		//iterate through stored Q-values to find a match for the input action
		for(QValue q : qs){
			if(q.a.equals(a)){
				return q;
			}
		}
		
		throw new RuntimeException("Could not find matching Q-value.");
	}
	
	protected double maxQ(State s){
		if(this.tf.isTerminal(s)){
			return 0.;
		}
		List<QValue> qs = this.getQs(s);
		double max = Double.NEGATIVE_INFINITY;
		for(QValue q : qs){
			max = Math.max(q.q, max);
		}
		return max;
	}

	@Override
	public void planFromState(State initialState) {
		throw new UnsupportedOperationException("We are not supporting planning for this tutorial.");
	}

	@Override
	public void resetPlannerResults() {
		this.qValues.clear();
	}
	
	
	
	public static void main(String [] args){
		long startTime = System.currentTimeMillis();
		
		GridWorldDomain gwd = new GridWorldDomain(3, 3);
		gwd.setMapToFourRooms();
		
		//only go in intended directon 80% of the time
		gwd.setProbSucceedTransitionDynamics(1);
		
		Domain domain = gwd.generateDomain();
		
		//get initial state with agent in 0,0
		State s = GridWorldDomain.getOneAgentNoLocationState(domain);
		GridWorldDomain.setAgent(s, 0, 0);
		
		//terminate in top right corner
				TerminalFunction tf = new GridWorldTerminalFunction(10, 10);
		//all transitions return -1
		RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), 80, -1);
		
		//setup Q-learning with 0.99 discount factor, discrete state hashing factory, a value
		//function initialization that initializes all Q-values to value 0, a learning rate
		//of 0.1 and an epsilon value of 0.1.
		GridExplorerQLearnerWithoutPlotter ql = new GridExplorerQLearnerWithoutPlotter(domain, rf, tf, 0.99, new DiscreteStateHashFactory(), 
				new ValueFunctionInitialization.ConstantValueFunctionInitialization(1.), 
				0.1, 0.1);
		
		//run learning for 1000 episodes
		for(int i = 0; i < 1000; i++){
			EpisodeAnalysis ea = ql.runLearningEpisodeFrom(s);
			//System.out.println("Episode " + i + " took " + ea.numTimeSteps() + " steps.");
		}
		
		//get the greedy policy from it
		Policy p = new GreedyQPolicy(ql);
		
		//evaluate the policy with one roll out and print out the action sequence
		EpisodeAnalysis ea = p.evaluateBehavior(s, rf, tf);
		long finishTime = System.currentTimeMillis();
		System.out.println("QLearning applied : ");
        System.out.println("total time taken : " + (finishTime - startTime) + " ms");
		//System.out.println(ea.getActionSequenceString("\n"));
		System.out.println("max time steps : " + ea.maxTimeStep());
		System.out.println("num time steps : " +ea.numTimeSteps());

		//ql.valueFunctionVisualize((QComputablePlanner)ql, p, s);
	}
	
	public void valueFunctionVisualize(QComputablePlanner planner, Policy p, State s) {
		List<State> allStates = StateReachability.getReachableStates(
				s, (SADomain) domain, hashingFactory);
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT,
				GridWorldDomain.ATTX, GridWorldDomain.CLASSAGENT,
				GridWorldDomain.ATTY);

		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT,
				GridWorldDomain.ATTX, GridWorldDomain.CLASSAGENT,
				GridWorldDomain.ATTY);
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONNORTH,
				new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONSOUTH,
				new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONEAST,
				new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONWEST,
				new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphRenderStyle.DISTSCALED);

		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(
				allStates, svp, planner);
		gui.setSpp(spp);
		gui.setPolicy(p);
		gui.setBgColor(Color.GRAY);
		gui.initGUI();
	}

}