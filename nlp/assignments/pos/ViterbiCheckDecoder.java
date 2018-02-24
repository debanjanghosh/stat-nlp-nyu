package nlp.assignments.pos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import nlp.assignments.pos.POSTaggerTester.Trellis;
import nlp.assignments.pos.POSTaggerTester.TrellisDecoder;
import nlp.util.Counter;


class ViterbiCheckDecoder<S> implements TrellisDecoder<S> 
{

	@Override
	public List<S> getBestPath(Trellis<S> trellis) 
	{
		// TODO Auto-generated method stub
		 List<S> states = new ArrayList<S>();
         S currentState = trellis.getStartState();
         Counter<S> viterbi = new Counter<S>();
         viterbi.setCount(currentState, 0);
         Map<S, S> backPointer = new HashMap<S, S>();
         backPointer.put(currentState, currentState);
         while (!currentState.equals(trellis.getEndState())) 
         {
                 Counter<S> forwardTransitions = trellis.getForwardTransitions(currentState);
                 recursive(trellis, viterbi, backPointer, forwardTransitions);
//               currentState = (S)forwardTransitions.keySet().iterator().next();
                 for (S forwardState : forwardTransitions.keySet()) 
                 {
                         currentState = forwardState;
                         recursive(trellis, viterbi, backPointer, trellis.getForwardTransitions(currentState));
                 }
         }
         
         S backState = trellis.getEndState();
         S startState = trellis.getStartState();
     	StringBuffer ret = new StringBuffer();
		
         while (!backState.equals(startState)) 
         {
                 states.add(0, backState);
                 ret.append(backState.toString()+" --> ");
                 backState = backPointer.get(backState);
//               System.out.println(backState);
         }
         states.add(0, startState);
         ret.append(startState);
  //       System.out.println(ret.toString());
         
         return states;
	}
	
	 void recursive(Trellis<S> trellis, Counter<S> viterbi,
             Map<S, S> backPointer, Counter<S> forwardTransitions) 
             {
     for (S forwardState : forwardTransitions.keySet()) 
     {
             Counter<S> backwardTransitions = trellis.getBackwardTransitions(forwardState);
             double max = Double.NEGATIVE_INFINITY;
             double temp = 0;//Double.NEGATIVE_INFINITY;
             S backwardMax = null;
             for (S backwardState : backwardTransitions.keySet()) 
             {
                     double temp1 = viterbi.containsKey(backwardState) ? viterbi.getCount(backwardState) : Double.NEGATIVE_INFINITY;
                     double temp2 = backwardTransitions.getCount(backwardState);
//                   temp = viterbi.getCount(backwardState) + backwardTransitions.getCount(backwardState);
                     temp = temp1 + temp2;
//                   System.out.println(backwardState + ", " + forwardState + ", " + temp1 + ", " + temp2);
                     if (temp > max)
                      {
                                     max = temp;
                                     backwardMax = backwardState;
                 		}
             }
             
             viterbi.setCount(forwardState, max);
             backPointer.put(forwardState, backwardMax);
//                   System.out.println(forwardState + ", " + backwardMax);
     }
}
	
}
