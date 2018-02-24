package nlp.assignments.pos;

import java.util.ArrayList;
import java.util.List;

import nlp.assignments.pos.POSTaggerTester.Trellis;
import nlp.assignments.pos.POSTaggerTester.TrellisDecoder;
import nlp.util.Counter;

class HMMDecoder<S> implements TrellisDecoder<S> 
{
	//an inefficient HMM - 
	public List<S> getBestPath(Trellis<S> trellis) 
	{
		List<S> states = new ArrayList<S>();
		S currentState = trellis.getStartState();
		states.add(currentState);
		while (!currentState.equals(trellis.getEndState())) 
		{
			Counter<S> transitions = trellis
					.getForwardTransitions(currentState);
			S nextState = transitions.argMax();
			states.add(nextState);
			currentState = nextState;
		}
		return states;
	}

	

   
}