import java.util.ArrayList;

import logist.agent.Agent;
import logist.plan.Action;
import logist.plan.Action.Move;
import logist.plan.Action.Pickup;
import logist.simulation.Vehicle;
import logist.task.Task;
import logist.task.TaskDistribution;
import logist.topology.Topology;
import logist.topology.Topology.City;

class QValueIteration{

	private int numCities;
	private int numMyStates;
	private int numMyActions;
	private ArrayList<City> cityMap;
	private ArrayList<ArrayList<myAction>> actionMap;
	private ArrayList<ArrayList<Double>> rewardTable;
	private ArrayList<ArrayList<ArrayList<Double>>> transitionProbabilityTable;
	private ArrayList<Double> stateValueList;
	private ArrayList<Integer> policy; //map from myState index to action index
	
	private class myState {
		public City currentCity;
		public City taskCity;
		public int id;
		
		public myState(City current, City next) {
			this.currentCity = current;
			this.taskCity = next;
			int taskID;
			if(taskCity==null) {
				taskID = numCities;
			}
			else {
				taskID = this.taskCity.id;
			}
			this.id = currentCity.id + taskID*numCities;
		}
		
		public myState(int id) {
			int currentID = id%numCities;
			int taskID = (id-currentID)/numCities;
			this.currentCity = cityMap.get(currentID);
			this.taskCity = cityMap.get(taskID);
			this.id = id;
		}
		
	}
	
	private class myAction {
		public Boolean takeTask;
		public City nextCity;
		public int id;
		
		public myAction(Boolean take, City next) {
			this.takeTask = take;
			this.nextCity = next;
			if(takeTask) {
				this.id = nextCity.id + numCities;
			}
			else {
				this.id = nextCity.id;
			}
		}
		
		public myAction(int id) {
			int cityID = id%numCities;
			this.takeTask = (id/numCities)==1;
			this.nextCity = cityMap.get(cityID);
		}
	}
	
	private ArrayList<myAction> getActions(myState s){
		if(actionMap.get(s.id)!=null) {
			return actionMap.get(s.id);
		}
		else {
			ArrayList<myAction> actions = new ArrayList<myAction>();
			if(!(s.taskCity==null)) {
				actions.add(new myAction(true, s.taskCity));
			}
			for(City c:s.currentCity.neighbors()) {
				actions.add(new myAction(false, c));
			}
			actionMap.set(s.id, actions);
			return actions;
		}
	}
	
	private void buildRewardTable(Topology topology, TaskDistribution td) {
		this.rewardTable = new ArrayList<ArrayList<Double>>(this.numMyStates);
		for(int i=0; i<this.numMyStates; i++) {
			myState s = new myState(i);
			ArrayList<Double> actionList = new ArrayList<Double>(this.numMyActions);
			for(int j=0; j<this.numMyActions; j++) {
				actionList.add(null);
			}
			for(myAction action: this.getActions(s)) {
				int aID = action.id;
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
		this.transitionProbabilityTable = new ArrayList<ArrayList<ArrayList<Double>>>(this.numMyStates);
		for(int i=0; i<this.numMyStates; i++) {
			myState s = new myState(i);
			ArrayList<ArrayList<Double>> actionList = new ArrayList<ArrayList<Double>>(this.numMyActions);
			for(int j=0; j<this.numMyActions; j++) {
				actionList.add(null);
			}
			for(myAction action: this.getActions(s)) {
				int aID = action.id;
				ArrayList<Double> nextmyStateList = new ArrayList<Double>(this.numMyStates);
				for(int k=0; k<this.numMyStates; k++) {
					nextmyStateList.add(0.0);
				}
				double non_null_prob = 0.0;
				for(City taskTarget: topology) {
					double probability = td.probability(action.nextCity, taskTarget);
					myState nextmyState = new myState(action.nextCity, taskTarget);
					nextmyStateList.set(nextmyState.id, probability);
					non_null_prob += probability;
				}
				// add null probability
				myState nullmyState = new myState(action.nextCity, null);
				nextmyStateList.set(nullmyState.id, 1.0-non_null_prob);
				
				actionList.set(aID, nextmyStateList);
			}
			this.transitionProbabilityTable.add(actionList);
		}
	}
	
	public Action optimalAction(Vehicle vehicle, Task availableTask) {
		Action action;
		City currentCity = vehicle.getCurrentCity();
		City nextCity;
		if(availableTask==null) {
			nextCity = null;
		}
		else {
			nextCity = availableTask.deliveryCity;
		}
		
		myState currentState = new myState(currentCity, nextCity);
		myAction optimalAction = new myAction(this.policy.get(currentState.id));
		if(optimalAction.takeTask) {
			action = new Pickup(availableTask);
		}
		else {
			action = new Move(optimalAction.nextCity);
		}
		return action;
	}
	
	// run the value iteration to determine optimal policy
	private void valueIteration(double discount, double threshold) {
		ArrayList<Double> myStateValueCopy = new ArrayList<Double>(this.numMyStates);
		for(int i=0; i<this.numMyStates; i++) {
			myStateValueCopy.add((double) this.stateValueList.get(i));
		}
		double delta = 100.0;
		do {
			for(int i=0; i<this.numMyStates; i++) {
				double maxValue = -1000000000000000.0;
				myState s = new myState(i);
				for(myAction a: this.getActions(s)) {
					double qValue = this.rewardTable.get(i).get(a.id);
					for(int k=0; k<this.numMyStates; k++) {
						double transProb = this.transitionProbabilityTable.get(i).get(a.id).get(k);
						qValue+=discount*(transProb*myStateValueCopy.get(k));
					}
					if(qValue>maxValue) {
						maxValue = qValue;
						this.policy.set(i, a.id);
					}
				}
				myStateValueCopy.set(i, maxValue);
			}
			// compute delta change in value list
			delta = 0.0;
			for(int i=0; i<this.numMyStates; i++) {
				double vsDelta = this.stateValueList.get(i) - myStateValueCopy.get(i);
				this.stateValueList.set(i, (double) myStateValueCopy.get(i));
				delta+= vsDelta*vsDelta;
			}
		} while(delta>threshold);
		System.gc();
	}
	
	// Constructor
	public QValueIteration(Topology topology, TaskDistribution td, Agent agent) {
		// construct helper variables for efficiency
		this.numCities = topology.size();
		this.numMyActions = topology.size()*2;
		this.numMyStates = topology.size()*(topology.size()+1);
		// map from city id to city
		this.cityMap = new ArrayList<City>(this.numCities);
		for(City c: topology) {
			this.cityMap.add(c.id, c);
		}
		this.cityMap.add(null);
		
		// map from myState id to list of possible actions
		this.actionMap = new ArrayList<ArrayList<myAction>>(this.numMyStates);
		for(int i=0; i<this.numMyStates; i++) {
			this.actionMap.add(i, null);
		}
		
		// build reward table for rapid lookup
		this.buildRewardTable(topology, td);
		
		// build transition probability table
		this.buildTransitionProbabilityTable(topology, td);
		
		// initialize value and policy arrays
		this.stateValueList = new ArrayList<Double>(this.numMyStates);
		this.policy = new ArrayList<Integer>(this.numMyStates);
		for(int i=0; i<this.numMyStates; i++) {
			this.stateValueList.add(0.0);
			this.policy.add(-1);
		}
		
		// Reads the discount factor from the agents.xml file.
		// If the property is not present it defaults to 0.95
		Double discount = agent.readProperty("discount-factor", Double.class,
				0.95);

		// Reads the iteration threshold from the agents.xml file.
		// If the property is not present it defaults to 0.00000000000001
		Double threshold = agent.readProperty("threshold", Double.class,
				0.00000000000001);
		
		// run value iteration algorithm to determine optimal policy
		this.valueIteration(discount, threshold);
	}
}