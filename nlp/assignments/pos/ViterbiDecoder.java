package nlp.assignments.pos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import nlp.assignments.pos.POSTaggerTester.Trellis;
import nlp.assignments.pos.POSTaggerTester.TrellisDecoder;
import nlp.util.Counter;


class ViterbiDecoder<S> implements TrellisDecoder<S> 
{
	//viterbi decoder - 
	
	public List<S> getBestPath(Trellis<S> trellis) 
	{
		List<S> states = new ArrayList<S>();
		S currentState = trellis.getStartState();
//		states.add(currentState);
		
		//we need to keep the max score for each state
		Counter<S> viterbiMax = new Counter<S>();
		viterbiMax.setCount(currentState, 0);
		
		//we also need to keep the state-to-state map that gave the "max" probability till this point
		Map<S,S> statePerStateMaxMap = new HashMap<S,S>();
		//init with current state
		statePerStateMaxMap.put(currentState, currentState);
        
		while (!currentState.equals(trellis.getEndState())) 
		{
			//for t = 1 we need to calculate separately so that
			//we use <S> or the zeroth state
			Counter<S> forwardTransitions = trellis
					.getForwardTransitions(currentState);
		
			//this is t=1
			recursion(trellis,forwardTransitions,viterbiMax,statePerStateMaxMap);
			
			//now do t = 2...n
			for ( S fwdState : forwardTransitions.keySet())
			{
				currentState = fwdState ;
				forwardTransitions = trellis
						.getForwardTransitions(currentState);
				
				recursion(trellis,forwardTransitions,viterbiMax,statePerStateMaxMap);
				
			}
			
			
		}
		
		//now backtrack all states till we hit the first state
		S lastState = trellis.getEndState();
		S firstState = trellis.getStartState();
		
		StringBuffer ret = new StringBuffer();
		while (!lastState.equals(firstState) )
		{
			states.add(0,lastState);
			ret.append(lastState.toString()+" --> ");
			lastState = statePerStateMaxMap.get(lastState);
		
		}
		states.add(0,firstState);
		ret.append(firstState.toString());
//		System.out.println(ret);
		return states;
	}
	
	private void recursion(Trellis<S> trellis,
			Counter<S> forwardTransitions, Counter<S> viterbiMax,
			Map<S, S> statePerStateMaxMap) 
	{
		// TODO Auto-generated method stub
		for ( S fwdState : forwardTransitions.keySet() )
		{
				Counter<S> backwardStates =	trellis.getBackwardTransitions(fwdState);
				
				double localMaxScore = Double.NEGATIVE_INFINITY ;
				S maxBwdState = null ;
				
				for ( S bwdState : backwardStates.keySet())
				{
					//we need two scores (previous max and current emission/transmission)
					double score1 = backwardStates.getCount(bwdState);
				    double score2 = viterbiMax.containsKey(bwdState) ? viterbiMax.getCount(bwdState) : Double.NEGATIVE_INFINITY;
	                   
					
					double totalScore = score1+score2 ;
					if ( totalScore > localMaxScore )
					{
						localMaxScore = totalScore ;
						maxBwdState = bwdState ;
					}
				}
				statePerStateMaxMap.put(fwdState, maxBwdState);
				viterbiMax.setCount(fwdState,localMaxScore);
		}
	}
}
