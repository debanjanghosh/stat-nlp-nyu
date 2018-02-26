package nlp.project;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import nlp.assignments.mt.WordAlignmentTester.SentencePair;
//import nlp.assignments.mt.WordAlignmentTester.Alignment;
//import nlp.assignments.mt.WordAlignmentTester.SentencePair;
//import nlp.assignments.mt.WordAlignmentTester.WordAligner;
import nlp.project.ParaphraseAlignmentTester.ParaphraseWordAligner;
import nlp.project.ParaphraseAlignmentTester.TRAIN_TYPE;
import nlp.util.Counter;
import nlp.util.CounterMap;
import nlp.util.Pair;


public class IBMModelTwo implements ParaphraseWordAligner 
{

	 public static final String NULL_WORD = "<NULL>";
	   private CounterMap<String, String> counter; // stores the fraction counts
	   private Counter<String> ecounter;
	   private Counter<String> fcounter;
	private CounterMap<String, String> fraction;
	private double INIT_PROB;
	   private static int NULL_POSITION = -1;
	   public static double NULL_PROB = 0.3;
	   private static int EM_ITERATION = 20;
	   
	   private static double alpha = 0.5 ;
	   
	//   private HeuristicWordAligner heuristicWordALigner ; 
	   
		Counter<String> englishCounter = new Counter<String>();
		Counter<String> frenchCounter = new Counter<String>();
		private List<ParaphraseSentencePair> trainingPairs;
		
		private double[][][][] expAlignProbs ;
		
		private void setExpAlignProbs()
		{
			expAlignProbs = new double[1000][1000][1000][1000] ;
			
			for ( int i = 0 ; i < 1000 ; i++ )
			{
				for ( int j = 0 ; j < 1000 ; j++ )
				{
					for ( int k = 0 ; k < 1000 ; k++ )
					{
						for ( int l = 0 ; l < 1000 ; l++ )
						{
							expAlignProbs[i][j][k][l] = -1;
						}
					}
				}
				
			}
			
		}
		

	   public Double getExpAlignProb(int fr_pos, int en_pos, int fr_len, int en_len)
	   {
			 double sum_so_far = 0.0;
			 double numer = 0.0;
			 
			 
			 
		//	 if(expAlignProbs[fr_pos][en_pos][fr_len][en_len] !=-1)
		//	 {
		//		 return expAlignProbs[fr_pos][en_pos][fr_len][en_len] ;
		//	 }
			 
			  
			  for (int i=0; i<en_len; i+=1)
			  {
				  Double term = Math.exp(-alpha*Math.abs(fr_pos - i*((double) fr_len)/((double) en_len)));
				  if (i == en_pos)
					  numer = term;
				  sum_so_far += term;
			  }
			  
			  double value =   (1-NULL_PROB ) * (numer/sum_so_far);
			  return value ;
		//	  expAlignProbs[fr_pos][en_pos][fr_len][en_len] = value ;
		//	  return expAlignProbs[fr_pos][en_pos][fr_len][en_len] ;
			  
			  
			  
		//	  return  (1-0 ) * (numer/sum_so_far);
		  }
	   public ParaphraseAlignment alignSentencePair(ParaphraseSentencePair sentencePair) 
	   {
		   
		   ArrayList<String> firstWordList = new ArrayList<String>();
		   ArrayList<String> secondWordList = new ArrayList<String>();
		   
	
		   
		   ParaphraseAlignment alignment = new ParaphraseAlignment();
	      List<String> FrenchWords = sentencePair.getFrenchWords();
	      List<String> EnglishWords = new ArrayList<String>(sentencePair.getEnglishWords());
	//      double UNI_PROB = (1.0 - NULL_PROB) / EnglishWords.size();
	      EnglishWords.add(NULL_WORD);
	      int fIndex = 0 ;
	      for (String fword : FrenchWords) 
	      {
	         double prob_max = 0.0;
	         double dice_max  = 0.0 ;
	         int max_pos_prob = NULL_POSITION;
	         int max_pos_dice = NULL_POSITION;
	         
	         int eIndex = 0 ;
	         for (String eword : EnglishWords) 
	         {
	        	 Counter<String> ecounts = counter.getCounter(eword);
	        	 Counter<String> fcounts = counter.getCounter(fword);
		        	
	        	
	            double prob_e_f = counter.getCount(eword, fword);
	            //double dice = getDiceCoeffn(eword,fword);
	            
	      //      double expProb = ;
	            
	       //     prob_e_f *= eword.equals(NULL_WORD) ? NULL_PROB : UNI_PROB; 
	            if(eword.equalsIgnoreCase(fword))
	            {
	            	prob_e_f = 1.0 ;
	            }
	            else
	            {
	            
	            	prob_e_f *= eword.equals(NULL_WORD) ? NULL_PROB : getExpAlignProb(fIndex, eIndex, FrenchWords.size(), EnglishWords.size() ); 
	            }
	            
	            if (prob_e_f > prob_max) 
	            {
	               prob_max = prob_e_f;
	               max_pos_prob = eword.equals(NULL_WORD) ? NULL_POSITION : EnglishWords.indexOf(eword);
	            }
	    /*        
	            if (dice > dice_max) 
	            {
	            	dice_max = dice;
	               max_pos_dice = eword.equals(NULL_WORD) ? NULL_POSITION : EnglishWords.indexOf(eword);
	            }
	         */   
	            eIndex++ ;
	         }    
	       /*  
	         if ( max_pos_dice == max_pos_prob )
	         {
	        	 alignment.addAlignment(max_pos_prob, FrenchWords.indexOf(fword), true);
	         }
	        */ 
	        // else
	         {
	         	 alignment.addAlignment(max_pos_prob, FrenchWords.indexOf(fword), true);
	      	   
	         }
	         
	         String firstWord = "BLANK" ;
         	 if ( max_pos_prob != -1 )
         	 {
         		 firstWord = EnglishWords.get(max_pos_prob) ;
         	 }
         	 
      	     setWord(firstWordList,secondWordList,max_pos_prob,FrenchWords.indexOf(fword),
      	    		firstWord ,fword) ;
         

	         
	         fIndex++ ;
	      }
	                
	      System.out.println(printAlignedList(firstWordList,secondWordList));
		   
	      
	      return alignment;
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
				List<String> englishWords = pair.getEnglishWords();
				List<String> frenchWords = pair.getFrenchWords();

				if (englishWords.contains(englishWord)
						&& frenchWords.contains(frenchWord)) {
					C++;
				}
				if (!englishPresent && englishWords.contains(englishWord)) {
					englishCounter.incrementCount(englishWord, 1.0);
					// A++ ;
				}
				if (!frenchPresent && frenchWords.contains(frenchWord)) {
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
	                                  List<String> sourceSentence, ParaphraseAlignment alignment) {
	      return 0;
	   }

	   public CounterMap<String, String> getProbSourceGivenTarget() 
	   {
	      return counter;
	   }

	   public void train(List<ParaphraseSentencePair> trainingPairs) 
	   {
		//   setExpAlignProbs() ;
	//	   heuristicWordALigner = new HeuristicWordAligner() ;
		   
		  Set<String> frenchWords = new HashSet<String>() ;
		  
	      fraction = new CounterMap<String, String>();
	      ecounter = new Counter<String>();
	      fcounter = new Counter<String>();
	      
	      this.trainingPairs = trainingPairs ;
	      
	      for (ParaphraseSentencePair pair : trainingPairs) 
	      {
	         List<String> eWords = pair.getEnglishWords();
	         List<String> fWords = pair.getFrenchWords();
	         for (String fword : fWords)
	         {
	      //      fcounter.incrementCount(fword, 1.0);
	            frenchWords.add(fword);
	         }
	       //  for (String eword : eWords)
	       //     ecounter.incrementCount(eword, 1.0);
	      }
	 //     ecounter.incrementCount(NULL_WORD, 1.0);

	      // Initialize with equal weights
	      INIT_PROB = 1.0 / frenchWords.size();
	      
	      
	//      for (String eword : ecounter.keySet())
	 //        for (String fword : fcounter.keySet())
	  //          fraction.setCount(eword, fword, INIT_PROB);

	      
	      
	      // EM
	      int itr = 1 ;
	      
	      for (int i = 0; i < EM_ITERATION; i++) 
	      {
	         //System.err.println(fraction.toString());
	         CounterMap<String, String> newfraction = new CounterMap<String, String>();
	         // E-step
	         for (ParaphraseSentencePair pair : trainingPairs) 
	         {
	            List<String> eWords = new ArrayList<String>(pair.getEnglishWords());
	        //    double UNI_PROB = (1.0 - NULL_PROB) / eWords.size();
	            eWords.add(NULL_WORD);
	            List<String> fWords = pair.getFrenchWords();
	            int fIndex = 0 ;
	            for (String fword : fWords) 
	            {
	               double sum_prob = 0.0;
	               int eIndex = 0 ;
	               for (String eword : eWords) 
	               {
	                  if (eword.equals(NULL_WORD)) sum_prob += NULL_PROB * getFractionProb(eword, fword);
	                  else sum_prob += getExpAlignProb(fIndex, eIndex, fWords.size(), eWords.size() ) * getFractionProb(eword, fword);
	               
	                  eIndex++ ;
	               }
	               eIndex = 0 ;
	               for (String eword : eWords) 
	               {
	                  if (eword.equals(NULL_WORD)) newfraction.incrementCount(eword, fword, NULL_PROB * getFractionProb(eword, fword) / sum_prob);
	                  else newfraction.incrementCount(eword, fword, getExpAlignProb(fIndex, eIndex, fWords.size(), eWords.size() ) * getFractionProb(eword, fword) / sum_prob);
	                  
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
	            //total_fraction /= eword.equals(NULL_WORD) ? NULL_PROB : UNI_PROB;
	            if (total_fraction > 1e-6) 
	            {
	               for (String fword : ewordcounter.keySet()) 
	               {
	             //     newfraction.setCount(eword, fword, newfraction.getCount(eword, fword) / total_fraction);
	            	   double norm = (newfraction.getCount(eword, fword)   ) /(total_fraction ) ;
	                  newfraction.setCount(eword, fword, norm);
		               
	               }
	            }
	         }
	         fraction = newfraction;
	         
	         System.out.println("EM ITERATION # IS " + itr) ;
	         itr++ ;
	      }
	      counter = fraction;
	   }
	        
	   public CounterMap<String, String>getCounter() {
	      return counter;
	   }
	   private double getFractionProb(String count1, String count2) 
	   {
		// TODO Auto-generated method stub
		 double val = fraction.getCount(count1, count2) ;
		 if ( val == 0)
		 {
			 return INIT_PROB ;
		 }
		return val;
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


	@Override
	public ParaphraseAlignment alignPhrasePair(
			ParaphraseSentencePair sentencePair) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public void normalizePhraseTable() {
		// TODO Auto-generated method stub
		
	}
	}
