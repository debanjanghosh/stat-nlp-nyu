package nlp.project;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import nlp.assignments.mt.HeuristicWordAligner;
import nlp.assignments.mt.WordAlignmentTester.Alignment;
import nlp.assignments.mt.WordAlignmentTester.SentencePair;
import nlp.project.ParaphraseAlignmentTester.ParaphraseWordAligner;
import nlp.project.ParaphraseAlignmentTester.TRAIN_TYPE;
import nlp.util.Counter;
import nlp.util.CounterMap;
import nlp.util.GeneralPriorityQueue;
import nlp.util.Pair;

public class IBMModelTwoWithAgreementGPQ implements ParaphraseWordAligner 
{

	public enum TYPE
	{
		ENG_FREN, FREN_ENG 
	}
	
	public static final String NULL_WORD = "<NULL>";
	
	private CounterMap<String,String> finalPhraseTable = new CounterMap<String,String>() ;
	
	private CounterMap<String, String> counter_ef; // stores the fraction counts
	private CounterMap<String, String> counter_fe; // stores the fraction counts
	
	
	private Counter<String> ecounter;
	private Counter<String> fcounter;
	
//	private CounterMap<String, String> fraction;
	
	private double INIT_PROB;
	private static int NULL_POSITION = -1;
	public static double NULL_PROB = 0.3;
	private static int EM_ITERATION = 20;

	private static double alpha = 0.2;

	private HeuristicWordAligner heuristicWordALigner;

	Counter<String> englishCounter = new Counter<String>();
	Counter<String> frenchCounter = new Counter<String>();
	private List<ParaphraseSentencePair> trainingPairs;

	public Double getExpAlignProb(int fr_pos, int en_pos, int fr_len, int en_len) 
	{

		double sum_so_far = 0.0;
		double numer = 0.0;

		for (int i = 0; i < en_len; i += 1)
		{
			Double term = Math.exp(-alpha
					* Math.abs(fr_pos - i * ((double) fr_len)
							/ ((double) en_len)));
			if (i == en_pos)
				numer = term;
			sum_so_far += term;
		}

		  double value =   (1-NULL_PROB ) * (numer/sum_so_far);
		  return value ;
	}
	
	private ParaphraseAlignment intersection ( GeneralPriorityQueue<String> efQueueObj , 
			GeneralPriorityQueue<String> feQueueObj, List<String> firstWords, List<String> secondWords  )
	{
		  ArrayList<String> firstWordList = new ArrayList<String>();
		  ArrayList<String> secondWordList = new ArrayList<String>();
	
		ParaphraseAlignment IntersectionAlignment = new ParaphraseAlignment();
		
		 while(efQueueObj.hasNext())
	      {
	    	  String topEF = efQueueObj.next();
	    	  
	    	  Double presence = feQueueObj.getPriority(topEF) ;
	    	  
	    	  if ( presence.doubleValue() == 0 || presence.isNaN() || presence.isInfinite())
	    	  {
	    		  
	    	  }
	    	  else
	    	  {
	    		  String features[] = topEF.split("\\|") ;
	    		  int english = Integer.valueOf(features[0]);
	    		  int french = Integer.valueOf(features[1]);
	    		  IntersectionAlignment.addAlignment(english,french, true);
	    	      
	    	      setWord(firstWordList,secondWordList,english,french,
	    	    		  firstWords.get(english) , secondWords.get(french)) ;
	    	  }
	      }
	      
	      System.out.println(printAlignedList(firstWordList,secondWordList));
	     
	      return IntersectionAlignment ;
	}
	
	
	
	
	 public ParaphraseAlignment alignPhrasePair(ParaphraseSentencePair sentencePair) 
	 {
		  List<String> secondWords = sentencePair.getSecondWords();
	      List<String> firstWords = sentencePair.getFirstWords();
	      
	      if(secondWords.contains("arrogant") || firstWords.contains("arrogant"))
	      {
	    	  System.out.println("here");
	      }
	  	  
	      //e2f
	      Pair<GeneralPriorityQueue<String>,ParaphraseAlignment> efQueueAlignmentObj = 
	    		  align(firstWords,secondWords,TYPE.ENG_FREN);  
	      GeneralPriorityQueue<String> efQueueObj = efQueueAlignmentObj.getFirst().deepCopy() ;
	      ParaphraseAlignment e2fAlignment = efQueueAlignmentObj.getSecond();
	      
	      //f2e
	      Pair<GeneralPriorityQueue<String>,ParaphraseAlignment> feQueueAlignmentObj =  
	    		  align(secondWords,firstWords,TYPE.FREN_ENG);  
	      
	      GeneralPriorityQueue<String> feQueueObj = feQueueAlignmentObj.getFirst().deepCopy() ;
	      ParaphraseAlignment f2eAlignment = feQueueAlignmentObj.getSecond();
	     
	      
	      //intersection
	      ParaphraseAlignment IntersectionAlignment = intersection(efQueueObj,feQueueObj,firstWords,secondWords) ;
	      efQueueObj = efQueueAlignmentObj.getFirst() ;
	      feQueueObj = feQueueAlignmentObj.getFirst() ;
	
	      //union 
	      ParaphraseAlignment unionAlignment = union(efQueueObj,feQueueObj,firstWords,secondWords) ;
	      efQueueObj = efQueueAlignmentObj.getFirst() ;
	      feQueueObj = feQueueAlignmentObj.getFirst() ;
	
	  	
	      //now we generate diagonal phrases from the alignment
	      generatePhraseDiagonals(e2fAlignment,f2eAlignment,IntersectionAlignment,unionAlignment,secondWords,firstWords);
	      
	      
	      return IntersectionAlignment;
	   }
	
	  public ParaphraseAlignment alignSentencePair(ParaphraseSentencePair sentencePair) 
	   {
		
		  ArrayList<String> firstWordList = new ArrayList<String>();
		  ArrayList<String> secondWordList = new ArrayList<String>();
		  
		  ParaphraseAlignment IntersectionAlignment = new ParaphraseAlignment();
	      List<String> secondWords = sentencePair.getSecondWords();
	      List<String> firstWords = sentencePair.getFirstWords();
	      
	      //e2f
	      Pair<GeneralPriorityQueue<String>,ParaphraseAlignment> efQueueAlignmentObj = 
	    		  align(firstWords,secondWords,TYPE.ENG_FREN);  
	      GeneralPriorityQueue<String> efQueueObj = efQueueAlignmentObj.getFirst() ;
	      ParaphraseAlignment e2fAlignment = efQueueAlignmentObj.getSecond();
	      
	      //f2e
	      Pair<GeneralPriorityQueue<String>,ParaphraseAlignment> feQueueAlignmentObj =  
	    		  align(secondWords,firstWords,TYPE.FREN_ENG);  
	      
	      GeneralPriorityQueue<String> feQueueObj = feQueueAlignmentObj.getFirst() ;
	      ParaphraseAlignment f2eAlignment = feQueueAlignmentObj.getSecond();
	  
	      while(efQueueObj.hasNext())
	      {
	    	  String topEF = efQueueObj.next();
	    	  
	    	  Double presence = feQueueObj.getPriority(topEF) ;
	    	  
	    	  if ( presence.doubleValue() == 0 || presence.isNaN() || presence.isInfinite())
	    	  {
	    		  
	    	  }
	    	  else
	    	  {
	    		  String features[] = topEF.split("\\|") ;
	    		  int english = Integer.valueOf(features[0]);
	    		  int french = Integer.valueOf(features[1]);
	    		  IntersectionAlignment.addAlignment(english,french, true);
	    	      
	    	      setWord(firstWordList,secondWordList,english,french,
	    	    		  firstWords.get(english) , secondWords.get(french)) ;
	    	  }
	      }
	      
	      System.out.println(printAlignedList(firstWordList,secondWordList));
	      
	       return IntersectionAlignment;
	   }


	  private void generatePhraseDiagonals(ParaphraseAlignment e2f,ParaphraseAlignment f2e,
			ParaphraseAlignment intersection, ParaphraseAlignment union, List<String> secondWords, List<String> firstWords) 
	  {
		// TODO Auto-generated method stub
		  //GROW DIAG
		  ParaphraseAlignment finalPhraseAlignment = GROW_DIAG(intersection,union,secondWords,firstWords);
		  
		  //FINAL 1 
		  finalPhraseAlignment = FINAL(e2f,finalPhraseAlignment,secondWords,firstWords, TYPE.ENG_FREN);
		  
		  //FINAL 2
		  finalPhraseAlignment = FINAL(f2e,finalPhraseAlignment,secondWords,firstWords, TYPE.FREN_ENG);
			 
		  //there are some versions of the above for union instead e2f f2e
		  finalPhraseAlignment = FINAL(union,finalPhraseAlignment,secondWords,firstWords, TYPE.ENG_FREN);
			
		  
		  //we need to print this final phrase alignment
		  Set<Pair<Integer,Integer>> sureAligns = finalPhraseAlignment.getSureAlignment() ;
		  //send this to a counterMap
		  
		  createPhraseTable(sureAligns,firstWords,secondWords) ;
	//	  createPhraseTable(sureAligns,secondWords,firstWords) ;
		  
		  
		  
		  for ( Pair<Integer,Integer> sa : sureAligns )
		  {
			  System.out.println(sa.getSecond() +"->" + sa.getFirst()) ;
		  }
		  
	  }
	  
	  public void normalizePhraseTable()
	  {
		  finalPhraseTable.normalize();
		  
		  for ( String left : finalPhraseTable.keySet())
		  {
			  Counter<String> rightPhrases = finalPhraseTable.getCounter(left);
			  
			  System.out.println(left + "\t" + rightPhrases.toString()) ;
		  }
		  
		  
	  }
	  
	  private void createPhraseTable(Set<Pair<Integer, Integer>> sureAligns,
			List<String> firstWords, List<String> secondWords) 
	  {
		// TODO Auto-generated method stub
		CounterMap<Integer,Integer> phraseTable = new CounterMap<Integer,Integer>() ;
		List<Integer> leftPosn = new ArrayList<Integer>();
		List<Integer> rightPosn = new ArrayList<Integer>();
		
		
		for ( Pair<Integer,Integer> sa : sureAligns )
		{
			phraseTable.incrementCount(sa.getSecond(), sa.getFirst(), 1.0);
		}
		
		for ( Integer left : phraseTable.keySet())
		{
			Counter<Integer> map = phraseTable.getCounter(left);
			
			for ( Integer right : map.keySet())
			{
			//	System.out.println(left +"->" + right) ;
				leftPosn.add(left);
				rightPosn.add(right);
			}
		}
	
		int i = 0 ;
	
		List<Integer> leftPhrasePosn = new ArrayList<Integer>();
		List<Integer> rightPhrasePosn = new ArrayList<Integer>();
	
		
		//this is for diagonals
		for (  i = 0 ; i < leftPosn.size() ; i++ )
		{
			int left_val = leftPosn.get(i);
			int right_val = rightPosn.get(i);
			
			if ( leftPhrasePosn.size() == 0  && rightPhrasePosn.size() == 0 )
			{
				//first element in the loop?
				leftPhrasePosn.add(left_val);
				rightPhrasePosn.add(right_val);
			}
			else
			{
				int commonSize = leftPhrasePosn.size() ;
				
				int old_left = leftPhrasePosn.get(commonSize-1);
				int old_right = rightPhrasePosn.get(commonSize-1);
				
				int left_diff = left_val - old_left ;
				int right_diff = right_val - old_right ;
				
				if ( (left_diff == 1 ) && (right_diff == 1))
				{
					leftPhrasePosn.add(left_val);
					rightPhrasePosn.add(right_val);
				}
				else
				{
					//no phrases?
					int left_size = leftPhrasePosn.size() ;
					if ( left_size < 2)
					{
						//no phrase created - clear out
						leftPhrasePosn.clear();
						rightPhrasePosn.clear();
						
						//but insert the current one
						leftPhrasePosn.add(left_val);
						rightPhrasePosn.add(right_val);
						
					}
					else
					{
						//left size at least 2 so print 
						String leftPhrase = getPhrase(leftPhrasePosn,secondWords) ;
						String rightPhrase = getPhrase(rightPhrasePosn,firstWords) ;
						System.out.println("LEFT PHRASE IS :" + leftPhrase) ;
						System.out.println("RIGHT PHRASE IS :" + rightPhrase) ;
						
						leftPhrasePosn.clear();
						rightPhrasePosn.clear();
						
						finalPhraseTable.incrementCount(leftPhrase, rightPhrase, 1.0) ;
					}
				}
			}
		}
		//whatever we have - do print
		if ( leftPhrasePosn.size() > 1)
		{
			String leftPhrase = getPhrase(leftPhrasePosn,secondWords) ;
			String rightPhrase = getPhrase(rightPhrasePosn,firstWords) ;
			System.out.println("LEFT PHRASE IS :" + leftPhrase) ;
			System.out.println("RIGHT PHRASE IS :" + rightPhrase) ;
			
			leftPhrasePosn.clear();
			rightPhrasePosn.clear();
			
			finalPhraseTable.incrementCount(leftPhrase, rightPhrase, 1.0) ;
		}
		
		//from the counter map we can get easy horizontal ones 
		leftPhrasePosn.clear(); 
		rightPhrasePosn.clear() ;
		
		for ( Integer left : phraseTable.keySet())
		{
			Counter<Integer> map = phraseTable.getCounter(left);
			if ( map.size() > 1 )
			{
				//grow horizontal 
				leftPhrasePosn.add(left);
				for ( Integer right : map.keySet())
				{
					rightPhrasePosn.add(right);
				}
				
				String leftPhrase = getPhrase(leftPhrasePosn,secondWords) ;
				String rightPhrase = getPhrase(rightPhrasePosn,firstWords) ;
				System.out.println("LEFT PHRASE IS :" + leftPhrase) ;
				System.out.println("RIGHT PHRASE IS :" + rightPhrase) ;
				
				finalPhraseTable.incrementCount(leftPhrase, rightPhrase, 1.0) ;	
				leftPhrasePosn.clear();
				rightPhrasePosn.clear();
			}
			
			//print

			
			
		}
	}

	private String getPhrase(List<Integer> rightPhrasePosn,
			List<String> firstWords) 
	{
		// TODO Auto-generated method stub
		StringBuffer left = new StringBuffer();
	
		for ( Integer right : rightPhrasePosn)
		{
			left.append(firstWords.get(right));
			left.append(" ") ;
		}
		
		return left.toString();
	}

	private ParaphraseAlignment FINAL (ParaphraseAlignment ef,ParaphraseAlignment finalPhraseAlignment, 
			  List<String> secondWords, List<String> firstWords, TYPE type)
	  {
		  /*
		  FINAL(a):
			  for english word e-new = 0 ... en
			  for foreign word f-new = 0 ... fn
			  if ( ( e-new not aligned or f-new not aligned ) and
			  ( e-new, f-new ) in alignment a )
			  add alignment point ( e-new, f-new )
*/
		  
		  for ( int i = 0 ; i < firstWords.size() ; i++)
		  {
				String englishWord = firstWords.get(i);
				
				boolean english_new_align = checkAlignmentPresence(i,TYPE.ENG_FREN,finalPhraseAlignment) ;
				
				for ( int j = 0 ; j < secondWords.size() ; j++ )
				{
					String frenchWord = secondWords.get(j);
					
				//	if ( frenchWord.equalsIgnoreCase("is") && j == 8)
				//	{
				//		System.out.println("here");
				//	}
					
					boolean french_new_align = checkAlignmentPresence(j,TYPE.FREN_ENG,finalPhraseAlignment) ;
					
					if ( !english_new_align || !french_new_align)
					{
						boolean check_align = checkAlignment(i,j,type,ef);
						
						if (check_align)
						{
							finalPhraseAlignment.addAlignment(i, j, true);
						}
					}
					
				}
				
			}
		  
		  return finalPhraseAlignment ;
	  
	  }
	  
	  

	private ParaphraseAlignment GROW_DIAG(ParaphraseAlignment intersection,ParaphraseAlignment union,
			List<String> secondWords, List<String> firstWords) 
	{
		// TODO Auto-generated method stub
	/*	
		GROW-DIAG():
			iterate until no new points added
				for english word e = 0 ... en
					for foreign word f = 0 ... fn
						if ( e aligned with f )
							for each neighboring point ( e-new, f-new ):
								if ( ( e-new not aligned and f-new not aligned ) and
									( e-new, f-new ) in union( e2f, f2e ) )
									add alignment point ( e-new, f-new )
	 */
		int previousPointsAdded = 0 ;
		
		int currentAlignment = 0 ;
		
		ParaphraseAlignment newAlignment = intersection ;
		
		while ( true )
		{
			currentAlignment = 0 ;
			
			for ( int i = 0 ; i < firstWords.size() ; i++)
			{
				String englishWord = firstWords.get(i);
				
				for ( int j = 0 ; j < secondWords.size() ; j++ )
				{
					String frenchWord = secondWords.get(j);
					
					boolean align = checkAlignment(i,j,TYPE.ENG_FREN,intersection);
					
					if ( align)
					{
						//get the neighbours
						List<String> englighNeighbours = getNeighbours(firstWords,i) ; 
						List<String> frenchNeighbours = getNeighbours(secondWords,j ) ; 
						
						for ( int k = 0 ; k < englighNeighbours.size() ; k++ )
						{
							String english_new = englighNeighbours.get(k);
							int orig_first = firstWords.indexOf(english_new);
							
							boolean english_new_align = checkAlignmentPresence(orig_first,TYPE.ENG_FREN,intersection);
							
							for ( int l = 0 ; l < frenchNeighbours.size() ; l++ )
							{
								String french_new = frenchNeighbours.get(l) ;
								int orig_second = secondWords.indexOf(french_new);
								
								boolean french_new_align = checkAlignmentPresence(orig_second,TYPE.FREN_ENG,intersection);
								
								if ( !english_new_align && !french_new_align )
								{
									boolean union_present = checkAlignmentPresence(orig_first, TYPE.ENG_FREN, union) ;
									union_present = checkAlignmentPresence(orig_second, TYPE.FREN_ENG, union);
									
									//if union_present still true!
									if( union_present )
									{
										newAlignment.addAlignment(k,l, true) ;
										currentAlignment++ ;
									}
								}
								
							}
						}
						
						
						
						
					}
					
				}
			}
			//so no new alignment anymore!
			if( currentAlignment <1)
			{
				break ;
			}
			//previous == current
			if ( previousPointsAdded == currentAlignment)
			{
				break ;
			}
			previousPointsAdded = currentAlignment ;
		}
		
		return newAlignment ;
		
	}

	
	private ParaphraseAlignment union(GeneralPriorityQueue<String> efQueueObj, GeneralPriorityQueue<String> feQueueObj,
			List<String> firstWords, List<String> secondWords) 
	{
		// TODO Auto-generated method stub
		 ParaphraseAlignment unionAlignment = new ParaphraseAlignment();
		 
		  ArrayList<String> firstWordList = new ArrayList<String>();
		  ArrayList<String> secondWordList = new ArrayList<String>();
	
	     
		 //union
	      while(efQueueObj.hasNext())
	      {
	    	  String topEF = efQueueObj.next();
	    	  String features[] = topEF.split("\\|") ;
	    	  int english = Integer.valueOf(features[0]);
	    	  int french = Integer.valueOf(features[1]);
	    	  unionAlignment.addAlignment(english,french, true);	
	    	  
	    	  if (english == -1)
	    	  {
	    		  //set to the null
	    		  english = firstWords.size()-1 ;
	    	  }
	    	  
	    	  if (french == -1)
	    	  {
	    		  //set to the null
	    		  french = secondWords.size()-1 ;
	    	  }
	    	  
	    	  setWord(firstWordList,secondWordList,english,french,
    	    		  firstWords.get(english) , secondWords.get(french)) ;
	      }
	      //union
	      while(feQueueObj.hasNext())
	      {
	    	  String topEF = feQueueObj.next();
	    	  String features[] = topEF.split("\\|") ;
	    	  int english = Integer.valueOf(features[0]);
	    	  int french = Integer.valueOf(features[1]);
	    	  unionAlignment.addAlignment(english,french, true);	  
	    	  
	    	  if (english == -1)
	    	  {
	    		  //set to the null
	    		  english = firstWords.size()-1 ;
	    	  }
	    	  
	    	  if (french == -1)
	    	  {
	    		  //set to the null
	    		  french = secondWords.size()-1 ;
	    	  }
	    	  
	    	  
	      	  setWord(firstWordList,secondWordList,english,french,
    	    		  firstWords.get(english) , secondWords.get(french)) ;
	  
	    	  
	      }
	     
	      System.out.println(printAlignedList(firstWordList,secondWordList));
	     
		
	      return unionAlignment;
	}
	
	

	
	private Pair<GeneralPriorityQueue<String>,ParaphraseAlignment> align ( List<String> origFirstWords, List<String> origSecondWords, TYPE type )
	{
		
		 List<String> simpleVerbs = new ArrayList<String>() ;
		 simpleVerbs.add("be");
		 simpleVerbs.add("is");
		 simpleVerbs.add("are");
		 simpleVerbs.add("were");
		 
		 
		 GeneralPriorityQueue<String> queueObj = new  GeneralPriorityQueue<String>() ;
		 ParaphraseAlignment alignment = new ParaphraseAlignment();
		
		 List<String> firstWords = new ArrayList<String>();
		 firstWords.addAll(origFirstWords);
	     firstWords.add(NULL_WORD);
	      
	     List<String> secondWords = new ArrayList<String>();
	     secondWords.addAll(origSecondWords) ;
	      
	     for (int fIndex = 0; fIndex < secondWords.size() ;fIndex++) 
	      {
	    	 String fword = secondWords.get(fIndex);
	         double prob_max = 0.0;
	         double dice_max  = 0.0 ;
	         int max_pos_prob = NULL_POSITION;
	         int max_pos_dice = NULL_POSITION;
	         
	         for (int eIndex = 0 ; eIndex < firstWords.size() ; eIndex++) 
	         {
	        	 String eword = firstWords.get(eIndex);
	        	 double prob_e_f = 0 ;
	        	 if ( type.equals(TYPE.ENG_FREN))
	        	 {
	        		 Counter<String> temp = counter_ef.getCounter(eword);
	        		 prob_e_f = counter_ef.getCount(eword, fword);
	        	 }
	        	 if ( type.equals(TYPE.FREN_ENG))
	        	 {
	        		 Counter<String> temp = counter_ef.getCounter(eword);
		        	 prob_e_f = counter_fe.getCount(eword, fword);
	        	 }
	        //    prob_e_f *= eword.equals(NULL_WORD) ? NULL_PROB : UNI_PROB; 
	            double alignProb = getExpAlignProb(fIndex, eIndex, secondWords.size(), firstWords.size() ) ;
	         
	            if (eword.equalsIgnoreCase(fword))
	            {
	            	prob_e_f = 1.0;
	            }
	            else if (simpleVerbs.contains(fword) && simpleVerbs.contains(eword))
	            {
	            	prob_e_f = 1.0;
	            }
	            
	            else
	            {
		            
		            prob_e_f *= eword.equals(NULL_WORD) ? NULL_PROB : alignProb; 
	            }
	            
	            if (prob_e_f > prob_max) 
	            {
	               prob_max = prob_e_f;
	               max_pos_prob = eword.equals(NULL_WORD) ? NULL_POSITION : eIndex;
	            }
	         }    
	         
	         alignment.addAlignment(max_pos_prob, fIndex, true);
	         
	         if ( type.equals(TYPE.ENG_FREN))
	         {
		         queueObj.setPriority(max_pos_prob + "|" + fIndex, prob_max) ;
	         }
	         if ( type.equals(TYPE.FREN_ENG))
	         {
	        	 queueObj.setPriority (fIndex +"|" + max_pos_prob  , prob_max) ;
	         }
	         
	      }
	    
	     Pair<GeneralPriorityQueue<String>,ParaphraseAlignment> pairQueueAlign = new 
					Pair<GeneralPriorityQueue<String>,ParaphraseAlignment>(queueObj, alignment) ;
			
		 
		 return pairQueueAlign ;
	}
	/*

	private boolean checkAlignment(int k, String englishWord,
			ParaphraseAlignment alignment) 
	{
		// TODO Auto-generated method stub
		return false;
	}
*/
	private List<String> getNeighbours(List<String> words, int i) 
	{
		// TODO Auto-generated method stub
		int size = words.size() ;
		String present = words.get(i);
		List<String> neighbours = new ArrayList<String>() ;
		
		if ( i == 0  && size != 1) //only one word
		{
			neighbours.add(words.get(i+1));
		}
		else if ( i == (size-1) && size!=1 ) //last word
		{
			neighbours.add(words.get(i-1));
		}
		else
		{
			neighbours.add(words.get(i-1));
			neighbours.add(words.get(i+1));
		}
		return neighbours;
	}
	
	private boolean checkAlignmentPresence ( int i, TYPE type,ParaphraseAlignment alignment)
	{
		Set<Pair<Integer,Integer>> sureAligns =  alignment.getSureAlignment() ;
		
		for ( Pair<Integer,Integer> pair : sureAligns )
		{
			if ( type.equals(TYPE.ENG_FREN))
			{
				if (  (pair.getFirst().intValue() == i ) )
				{
					return true ;
				}
			}
			
			if ( type.equals(TYPE.FREN_ENG))
			{
				if (  (pair.getSecond().intValue() == i ) )
				{
					return true ;
				}
			}
		}
		
		return false;
	}

	private boolean checkAlignment(int i, int j, TYPE type, ParaphraseAlignment alignment) 
	{
		// TODO Auto-generated method stub
		Set<Pair<Integer,Integer>> sureAligns =  alignment.getSureAlignment() ;
	
		for ( Pair<Integer,Integer> pair : sureAligns )
		{
			if (type.equals(TYPE.ENG_FREN))
			{
				if (  (pair.getFirst().intValue() == i ) && ( pair.getSecond().intValue() == j) )
				{
					return true ;
				}
			}
			if (type.equals(TYPE.FREN_ENG))
			{
				if (  (pair.getFirst().intValue() == j ) && ( pair.getSecond().intValue() == i) )
				{
					return true ;
				}
			}
		}
		
		return false;
	}

	private List<String> printAlignedList(ArrayList<String> firstWordList,
				ArrayList<String> secondWordList) 
		{
			// TODO Auto-generated method stub
			List<String> ret = new ArrayList<String>();
			for ( int i = 0 ; i < secondWordList.size() ;i++)
			{
				ret.add( secondWordList.get(i) +"->" + firstWordList.get(i));
			}
			
			return ret;
		}
	   

	   private void setWord(ArrayList<String> firstWordList,
				ArrayList<String> secondWordList, int firstPosn, int secondPosn, String firstWord, String secondWord) 
		{
			// TODO Auto-generated method stub
			if (firstPosn != -1 &&  secondPosn != -1)
			{
				firstWordList.add(firstWord);
				secondWordList.add(secondWord);
			}
		}
	  
	  
	
	public double getDiceCoeffn(String englishWord, String frenchWord) 
	{
		// TODO Auto-generated method stub
		double C = 0;
		double A = 0;
		double B = 0;

		boolean englishPresent = englishCounter.containsKey(englishWord);
		boolean frenchPresent = frenchCounter.containsKey(frenchWord);

		for (ParaphraseSentencePair pair : trainingPairs) 
		{
			List<String> englishWords = pair.getFirstWords();
			List<String> frenchWords = pair.getSecondWords();

			if (englishWords.contains(englishWord)
					&& frenchWords.contains(frenchWord)) 
			{
				C++;
			}
			if (!englishPresent && englishWords.contains(englishWord)) 
			{
				englishCounter.incrementCount(englishWord, 1.0);
				// A++ ;
			}
			if (!frenchPresent && frenchWords.contains(frenchWord)) 
			{
				frenchCounter.incrementCount(frenchWord, 1.0);
				// B++ ;
			}
		}

		// Compute dice coef
		A = englishCounter.getCount(englishWord);
		B = frenchCounter.getCount(frenchWord);

		double dice = (2 * C) / (A + B);

		return dice;

	}

	public double getAlignmentProb(List<String> targetSentence,
			List<String> sourceSentence, Alignment alignment) {
		return 0;
	}
/*
	public CounterMap<String, String> getProbSourceGivenTarget() {
		return null;
	}
*/
	public List<ParaphraseSentencePair> invert(List<ParaphraseSentencePair> trainingPairs)
	{
		int index = 0 ;
		List<ParaphraseSentencePair> invertList = new ArrayList<ParaphraseSentencePair>();
		
		for ( ParaphraseSentencePair pair : trainingPairs)
		{
			List<String>englishWords = pair.getFirstWords() ;
			List<String>frenchWords = pair.getSecondWords() ;
			
			ParaphraseSentencePair newPair = new ParaphraseSentencePair(pair.getSentenceID(),pair.getTrainType(), frenchWords,englishWords);
			invertList.add(newPair);
		}
		return invertList ;
	}
	
	public void train(List<ParaphraseSentencePair> trainingPairs) 
	{
		//do regular training (english to french)
		counter_ef = new CounterMap<String, String>(); 	
		train(trainingPairs,TYPE.ENG_FREN) ;
		
		//now do french to english training
		List<ParaphraseSentencePair> invertPairs = invert(trainingPairs) ;
		counter_fe = new CounterMap<String, String>(); 
		train(invertPairs, TYPE.FREN_ENG) ;
	}
	
	public void train ( List<ParaphraseSentencePair> trainingPairs, TYPE type)
	{
//		heuristicWordALigner = new HeuristicWordAligner();

		Set<String> frenchWords = new HashSet<String>();

	//	CounterMap<String, String> fraction = new CounterMap<String, String>();
	
//		ecounter = new Counter<String>();
//		fcounter = new Counter<String>();

		this.trainingPairs = trainingPairs;

		for (ParaphraseSentencePair pair : trainingPairs) 
		{
			List<String> eWords = pair.getFirstWords();
			List<String> fWords = pair.getSecondWords();
			for (String fword : fWords) 
			{
				// fcounter.incrementCount(fword, 1.0);
				frenchWords.add(fword);
			}
			// for (String eword : eWords)
			// ecounter.incrementCount(eword, 1.0);
		}
		// ecounter.incrementCount(NULL_WORD, 1.0);

		// Initialize with equal weights
		INIT_PROB = 1.0 / frenchWords.size();
		frenchWords.clear();
		// for (String eword : ecounter.keySet())
		// for (String fword : fcounter.keySet())
		// fraction.setCount(eword, fword, INIT_PROB);

		// EM
		for (int i = 0; i < EM_ITERATION; i++) 
		{
			 System.err.println("THE EM ITERATION FOR "+ type.name() + " IS " +   (i+1));
			CounterMap<String, String> newfraction = new CounterMap<String, String>();
			// E-step
			for (ParaphraseSentencePair pair : trainingPairs) 
			{
				List<String> eWords = new ArrayList<String>(
						pair.getFirstWords());
		//		double UNI_PROB = (1.0 - NULL_PROB) / eWords.size();
				eWords.add(NULL_WORD);
				List<String> fWords = pair.getSecondWords();
				
				int fIndex = 0 ;
				for (String fword : fWords) 
				{
					double sum_prob = 0.0;
					int eIndex = 0 ;
					for (String eword : eWords) 
					{
						if (eword.equals(NULL_WORD))
							sum_prob += NULL_PROB
									* getFractionProb(type,eword, fword);
						else
							sum_prob += getExpAlignProb(fIndex, eIndex, fWords.size(), eWords.size() )
									* getFractionProb(type,eword, fword);
						
						eIndex++ ;
					}
					
					eIndex = 0 ;
					
					for (String eword : eWords) 
					{
						if (eword.equals(NULL_WORD))
							newfraction.incrementCount(eword, fword, NULL_PROB
									* getFractionProb(type,eword, fword) / sum_prob);
						else
							newfraction.incrementCount(eword, fword, getExpAlignProb(fIndex, eIndex, fWords.size(), eWords.size() )
									* getFractionProb(type,eword, fword) / sum_prob);
					
						eIndex++ ;
					
					}
					
					fIndex++ ;
				}
			}
			// M-step
			for (String eword : newfraction.keySet()) 
			{
				Counter<String> ewordcounter = newfraction.getCounter(eword);
				double total_fraction = ewordcounter.totalCount();
				// total_fraction /= eword.equals(NULL_WORD) ? NULL_PROB :
				// UNI_PROB;
				if (total_fraction > 1e-6) 
				{
					for (String fword : ewordcounter.keySet()) 
					{
						// newfraction.setCount(eword, fword,
						// newfraction.getCount(eword, fword) / total_fraction);
						double norm = (newfraction.getCount(eword, fword) + 2)
								/ (total_fraction + 1000);
						newfraction.setCount(eword, fword, norm);

					}
				}
			}
			if ( type.equals(TYPE.ENG_FREN))
			{
				counter_ef = newfraction;
			}
			if ( type.equals(TYPE.FREN_ENG))
			{
				counter_fe = newfraction;
			}
		}
	//	counter = fraction;
	}
/*
	public CounterMap<String, String> getCounter() 
	{
		return counter;
	}
*/
	private double getFractionProb(TYPE type, String count1, String count2) 
	{
		// TODO Auto-generated method stub
		double val  = 0;
		if ( type.equals(TYPE.ENG_FREN))
		{
			val = counter_ef.getCount(count1, count2);
			if (val == 0)
			{
				return INIT_PROB;
			}
			return val;
		}
		
		if ( type.equals(TYPE.FREN_ENG))
		{
			val = counter_fe.getCount(count1, count2);
			if (val == 0)
			{
				return INIT_PROB;
			}
			return val;
		}
		
		return val ;
	}

	@Override
	public void findBestPossibleAlignedPool(Pair<Integer, List<String>> original) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void train(List<ParaphraseSentencePair> trainingPPSentencePairs,
			TRAIN_TYPE paraphraseParaphrase) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setType(TRAIN_TYPE paraphraseParaphrase) {
		// TODO Auto-generated method stub
		
	}
}
