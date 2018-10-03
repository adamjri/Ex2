package template;

import java.util.Random;

import java.util.ArrayList;

import logist.simulation.Vehicle;
import logist.agent.Agent;
import logist.behavior.ReactiveBehavior;
import logist.plan.Action;
import logist.plan.Action.Move;
import logist.plan.Action.Pickup;
import logist.task.Task;
import logist.task.TaskDistribution;
import logist.topology.Topology;
import logist.topology.Topology.City;

public class ReactiveTemplate implements ReactiveBehavior {
	
	private int numCities;
	private int numStates;
	private int numMyActions;
	private ArrayList<City> cityMap;
	private ArrayList<ArrayList<myAction>> actionMap;
	private ArrayList<ArrayList<Double>> rewardTable;
	private ArrayList<ArrayList<ArrayList<Double>>> transitionProbabilityTable;
	private ArrayList<Double> stateValueList;
	private ArrayList<Integer> policy; //map from state index to action index
	
	private class State {
		public City currentCity;
		public City taskCity;
		
		public State(City current, City next) {
			this.currentCity = current;
			this.taskCity = next;
		}
		
		public State(int id) {
			int currentID = id%numCities;
			int taskID = (id-currentID)/numCities;
			this.currentCity = cityMap.get(currentID);
			this.taskCity = cityMap.get(taskID);
		}
		
		public int id() {
			int taskID;
			if(taskCity==null) {
				taskID = numCities;
			}
			else {
				taskID = this.taskCity.id;
			}
			return currentCity.id + taskID*numCities;
		}
		
	}
	
	private class myAction {
		public Boolean takeTask;
		public City nextCity;
		
		public myAction(Boolean take, City next) {
			this.takeTask = take;
			this.nextCity = next;
		}
		
		public myAction(int id) {
			int cityID = id%numCities;
			this.takeTask = (id/numCities)==1;
			this.nextCity = cityMap.get(cityID);
		}
		
		public int id() {
			if(takeTask) {
				return nextCity.id + numCities;
			}
			else {
				return nextCity.id;
			}
		}
	}
	
	private ArrayList<myAction> getActions(State s){
		if(actionMap.get(s.id())!=null) {
			return actionMap.get(s.id());
		}
		else {
			ArrayList<myAction> actions = new ArrayList<myAction>();
			if(!(s.taskCity==null)) {
				actions.add(new myAction(true, s.taskCity));
			}
			for(City c:s.currentCity.neighbors()) {
				actions.add(new myAction(false, c));
			}
			actionMap.set(s.id(), actions);
			return actions;
		}
	}
	
	private void buildRewardTable(Topology topology, TaskDistribution td) {
		this.rewardTable = new ArrayList<ArrayList<Double>>(this.numStates);
		for(int i=0; i<this.numStates; i++) {
			State s = new State(i);
			ArrayList<Double> actionList = new ArrayList<Double>(this.numMyActions);
			for(int j=0; j<this.numMyActions; j++) {
				actionList.add(null);
			}
			for(myAction action: this.getActions(s)) {
				int aID = action.id();
				City source = s.currentCity;
				City target = action.nextCity;
				double cost = td.weight(source, target)*source.distanceTo(target);
				double reward;
				if(action.takeTask) {
					reward = td.reward(source, target);
				}
				else {
					reward = 0.0;
				}
				actionList.set(aID, reward-cost);
			}
			this.rewardTable.add(actionList);
		}
	}
	
	private void buildTransitionProbabilityTable(Topology topology, TaskDistribution td) {
		this.transitionProbabilityTable = new ArrayList<ArrayList<ArrayList<Double>>>(this.numStates);
		for(int i=0; i<this.numStates; i++) {
			State s = new State(i);
			ArrayList<ArrayList<Double>> actionList = new ArrayList<ArrayList<Double>>(this.numMyActions);
			for(int j=0; j<this.numMyActions; j++) {
				actionList.add(null);
			}
			for(myAction action: this.getActions(s)) {
				int aID = action.id();
				ArrayList<Double> nextStateList = new ArrayList<Double>(this.numStates);
				for(int k=0; k<this.numStates; k++) {
					nextStateList.add(0.0);
				}
				double non_null_prob = 0.0;
				for(City taskTarget: topology) {
					double probability = td.probability(action.nextCity, taskTarget);
					State nextState = new State(action.nextCity, taskTarget);
					nextStateList.set(nextState.id(), probability);
					non_null_prob += probability;
				}
				// add null probability
				State nullState = new State(action.nextCity, null);
				nextStateList.set(nullState.id(), 1.0-non_null_prob);
				
				actionList.set(aID, nextStateList);
			}
			this.transitionProbabilityTable.add(actionList);
		}
	}
	
	private myAction policyMap(State s) {
		return new myAction(this.policy.get(s.id()));
	}
	
	// run the value iteration to determine optimal policy
	private void valueIteration(double discount, double threshold) {
		ArrayList<Double> stateValueCopy = new ArrayList<Double>(this.numStates);
		for(int i=0; i<this.numStates; i++) {
			stateValueCopy.add((double) this.stateValueList.get(i));
		}
		double delta = 100.0;
		do {
			for(int i=0; i<this.numStates; i++) {
				double maxValue = -1000000000000000.0;
				State s = new State(i);
				for(myAction a: this.getActions(s)) {
					double qValue = this.rewardTable.get(i).get(a.id());
					for(int k=0; k<this.numStates; k++) {
						double transProb = this.transitionProbabilityTable.get(i).get(a.id()).get(k);
						qValue+=discount*(transProb*stateValueCopy.get(k));
					}
					if(qValue>maxValue) {
						maxValue = qValue;
						this.policy.set(i, a.id());
					}
				}
				stateValueCopy.set(i, maxValue);
			}
			// compute delta change in value list
			delta = 0.0;
			for(int i=0; i<this.numStates; i++) {
				double vsDelta = this.stateValueList.get(i) - stateValueCopy.get(i);
				this.stateValueList.set(i, (double) stateValueCopy.get(i));
				delta+= vsDelta*vsDelta;
			}
		} while(delta>threshold);
		System.gc();
	}
	//******************************************************************************************************

	private Random random;
	private double pPickup;
	private int numActions;
	private Agent myAgent;

	@Override
	public void setup(Topology topology, TaskDistribution td, Agent agent) {
		// construct helper variables for efficiency
		this.numCities = topology.size();
		this.numMyActions = topology.size()*2;
		this.numStates = topology.size()*(topology.size()+1);
		// map from city id to city
		this.cityMap = new ArrayList<City>(this.numCities);
		for(City c: topology) {
			this.cityMap.add(c.id, c);
		}
		this.cityMap.add(null);
		
		// map from state id to list of possible actions
		this.actionMap = new ArrayList<ArrayList<myAction>>(this.numStates);
		for(int i=0; i<this.numStates; i++) {
			this.actionMap.add(i, null);
		}
		
		// build reward table for rapid lookup
		this.buildRewardTable(topology, td);
		
		// build transition probability table
		this.buildTransitionProbabilityTable(topology, td);
		
		// initialize value and policy arrays
		this.stateValueList = new ArrayList<Double>(this.numStates);
		this.policy = new ArrayList<Integer>(this.numStates);
		for(int i=0; i<this.numStates; i++) {
			this.stateValueList.add(0.0);
			this.policy.add(-1);
		}
		
		// Reads the discount factor from the agents.xml file.
		// If the property is not present it defaults to 0.95
		Double discount = agent.readProperty("discount-factor", Double.class,
				0.95);

		// run value iteration algorithm to determine optimal policy
		double threshold = 0.00000000000001;
		this.valueIteration(discount, threshold);

		this.random = new Random();
		this.pPickup = discount;
		this.numActions = 0;
		this.myAgent = agent;
	}

	@Override
	public Action act(Vehicle vehicle, Task availableTask) {
		Action action;
//
//		if (availableTask == null || random.nextDouble() > pPickup) {
//			City currentCity = vehicle.getCurrentCity();
//			action = new Move(currentCity.randomNeighbor(random));
//		} else {
//			action = new Pickup(availableTask);
//		}
//		
		City currentCity = vehicle.getCurrentCity();
		City nextCity;
		if(availableTask==null) {
			nextCity = null;
		}
		else {
			nextCity = availableTask.deliveryCity;
		}
		
		State currentState = new State(currentCity, nextCity);
		myAction optimalAction = this.policyMap(currentState);
		if(optimalAction.takeTask) {
			action = new Pickup(availableTask);
		}
		else {
			action = new Move(optimalAction.nextCity);
		}
		
		if (numActions >= 1) {
			System.out.println("The total profit after "+numActions+" actions is "+myAgent.getTotalProfit()+" (average profit: "+(myAgent.getTotalProfit() / (double)numActions)+")");
		}
		numActions++;
		
		return action;
	}
}
