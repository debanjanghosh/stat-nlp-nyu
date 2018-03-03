package nlp.assignments;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import nlp.util.Counter;
import nlp.util.TextUtil;

public class FeatureExtractorFunctions 
{

	private Set<String> companies;
	private Set<String> places;
	private Set<String> medicines;

	private Set<String> people;
	private HashSet<String> movies;
	private HashSet<String> placeEnds;
	private HashSet<String> countries;
	private HashSet<String> drugstart;
	private HashSet<String> stopWords;
	private HashSet<String> drugends;


	public FeatureExtractorFunctions() throws IOException
	{
		companies = new HashSet<String>();
		companies.add("corporation");
		companies.add("corp");
		companies.add("co.");
		
	//	companies.add("financing");
		
		
		companies.add("inc.");
		companies.add("limited");
		companies.add("ltd.");
		
		
		
		
		companies.add("trust");
		
		companies.add("l.p.");
		companies.add("llc");
		companies.add("companies");
		
		companies.add("corporation.");
		companies.add("income");
		companies.add("fund");
		companies.add("fund,");
		companies.add("group");
		companies.add("capital");
		companies.add("energy");
		companies.add("industries");
		companies.add("financial");
		companies.add("realty");
		companies.add("corp.");
		
		
		
		places = new HashSet<String>();
		places.add("bay");
		places.add("town");
		places.add("land");
		places.add("city");
		places.add("bridge");
		places.add("burge");
		places.add("field");
		places.add("borough");
		places.add("park");
		places.add("green");
	//	places.add("hill");
		places.add("hills");
		places.add("vale");
		places.add("port");
		places.add("castle");
		places.add("market");
		
		places.add("beach");
		
//		places.add("east");
//		places.add("west");
		places.add("point");
//		places.add("north");
//		places.add("south");
	//	places.add("trinidad");
	//	places.add("tobago");
		
	//	places.add("heights");
		
		
		medicines =new HashSet<String>();
		medicines.add("flu");
		medicines.add("cold");
		medicines.add("cough");
		medicines.add("allergy");
		medicines.add("sore");
		medicines.add("sinus");
		medicines.add("medicine");
		medicines.add("medicated");
		medicines.add("medication");
		medicines.add("antacid");
		
	//	medicines.add("acne");
	//	medicines.add("drug");
		
		medicines.add("headache");
		medicines.add("treatment") ;
		
		medicines.add("pills") ;
		medicines.add("botox") ;
		medicines.add("tablet");
		
		
		loadPersons();
		loadMedicines();
		loadMovies();
		loadPlaceEnds();
		loadCountries();
		loadDrugStart();
		loadDrugEnd();
		
		
		stopWords = new HashSet<String>();
		
	//	stopWords.add("the");
	//	stopWords.add("of");
	//	stopWords.add("and");
	//	stopWords.add("a");
		
	}
	
	public void loadStopWords() throws IOException
	{
		drugstart = new HashSet<String>();
		String file = "./data/data2/drug_start_1.txt";
		BufferedReader reader = new BufferedReader(new InputStreamReader (
					new FileInputStream( file ) , "UTF8") );
			
			//header
		int index = 0 ;
		while (true)
		{
			String line = reader.readLine();
			if (line == null)
			{
				break;
			}
				
			String features[] = line.split("\t");
			String person = features[0].trim();
			drugstart.add(person) ;
		}
		
		reader.close();
	}
	
	
	public void loadDrugEnd() throws IOException
	{
		drugends = new HashSet<String>();
		String file = "./data/data2/drug_end_1.txt";
		BufferedReader reader = new BufferedReader(new InputStreamReader (
					new FileInputStream( file ) , "UTF8") );
			
			//header
		int index = 0 ;
		while (true)
		{
			String line = reader.readLine();
			if (line == null)
			{
				break;
			}
				
			String features[] = line.split("\t");
			String person = features[0].trim();
			drugends.add(person) ;
		}
		
		reader.close();
	}
	public void loadDrugStart() throws IOException
	{
		drugstart = new HashSet<String>();
		String file = "./data/data2/drug_start_1.txt";
		BufferedReader reader = new BufferedReader(new InputStreamReader (
					new FileInputStream( file ) , "UTF8") );
			
			//header
		int index = 0 ;
		while (true)
		{
			String line = reader.readLine();
			if (line == null)
			{
				break;
			}
				
			String features[] = line.split("\t");
			String person = features[0].trim();
			drugstart.add(person) ;
		}
		
		reader.close();
	}
	
	public void loadCountries() throws IOException
	{
		countries = new HashSet<String>();
		String file = "./data/data2/country_1.txt";
		BufferedReader reader = new BufferedReader(new InputStreamReader (
					new FileInputStream( file ) , "UTF8") );
			
			//header
		int index = 0 ;
		while (true)
		{
			String line = reader.readLine();
			if (line == null)
			{
				break;
			}
				
			String features[] = line.split("\t");
			String person = features[0].trim();
			countries.add(person) ;
		}
		
		reader.close();
	}
	
	
	
	
	public void loadPlaceEnds() throws IOException
	{
		placeEnds = new HashSet<String>();
		String file = "./data/data2/placeends_1.txt";
		BufferedReader reader = new BufferedReader(new InputStreamReader (
					new FileInputStream( file ) , "UTF8") );
			
			//header
		int index = 0 ;
		while (true)
		{
			String line = reader.readLine();
			if (line == null)
			{
				break;
			}
				
			String features[] = line.split("\t");
			String person = features[0].trim();
			placeEnds.add(person) ;
		}
		
		reader.close();
	}
	
	
	
	public void loadMovies() throws IOException
	{
		movies = new HashSet<String>();
		String file = "./data/data2/movie_1.txt";
		BufferedReader reader = new BufferedReader(new InputStreamReader (
					new FileInputStream( file ) , "UTF8") );
			
			//header
		int index = 0 ;
		while (true)
		{
			String line = reader.readLine();
			if (line == null)
			{
				break;
			}
				
			String features[] = line.split("\t");
			String person = features[0].trim();
			movies.add(person) ;
		}
		
		reader.close();
	}
	
	public void loadMedicines() throws IOException
	{
		String file = "./data/data2/drug_1.txt";
		BufferedReader reader = new BufferedReader(new InputStreamReader (
					new FileInputStream( file ) , "UTF8") );
			
			//header
		int index = 0 ;
		while (true)
		{
			String line = reader.readLine();
			if (line == null)
			{
				break;
			}
				
			String features[] = line.split("\t");
			String person = features[0].trim();
			medicines.add(person) ;
		}
		
		reader.close();
	}
	
	public void loadPersons() throws IOException
	{
		people =  new HashSet<String>();
		
		String file = "./data/data2/person_1.txt";
		BufferedReader reader = new BufferedReader(new InputStreamReader (
					new FileInputStream( file ) , "UTF8") );
			
			//header
		int index = 0 ;
		while (true)
		{
			String line = reader.readLine();
			if (line == null)
			{
				break;
			}
				
			String features[] = line.split("\t");
			String person = features[0].trim();
			people.add(person) ;
		}
		
		reader.close();
	}
	
	
	public Counter<String> getNonAlphaFeatures (String name )
	{
		String tokens[] = name.split("\\s++") ;
		Counter<String> features = new Counter<String>();
		
		String ret = new String() ;
		
		boolean present = false ;
		for ( String token : tokens )
		{
			char[] chars = token.toCharArray();
			for ( char c : chars )
			{
				if ( ( 'a' <= c && c <= 'z') || ( 'A' <= c && c <= 'Z') || ( '0' <= c && c <= '9') )
				{
					
				}
				else
				{
					ret += c ;
					present = true ;
				}
			}
		}
		if ( present)
		{
			if ( ret.length() > 1)
				features.setCount("NONALPHA1-"+ ret, 1.0) ;
		}
		return features ;
	}
	
	public Counter<String> getWordFeatures (String name )
	{
		String tokens[] = name.split("\\s++") ;
		Counter<String> features = new Counter<String>();
		
		for ( int i = 0 ; i < tokens.length ; i++ )
		{
			String token = tokens[i];
			features.incrementCount("WORD1-"+token,1.0);
			
			int min = Math.min(token.length(), 4);
	//		features.incrementCount("PREFIX1-"+token.substring(0, min),1.0);
			
	//		features.incrementCount("SUFFIX1-"+token.substring(min,token.length()),1.0);
			
		}
		return features ;
	}
	
	public Counter<String> getBigramFeatures (String name )
	{
		String tokens[] = name.split("\\s++") ;
		Counter<String> features = new Counter<String>();
		
		String preToken = tokens[0].trim();
		for ( int i = 1 ; i < tokens.length ; i++ )
		{
			String token = tokens[i].trim();
			features.incrementCount("WORD2-"+preToken+token,1.0);
			
			int min = Math.min(token.length(), 4);
	//		features.incrementCount("PREFIX1-"+token.substring(0, min),1.0);
			
	//		features.incrementCount("SUFFIX1-"+token.substring(min,token.length()),1.0);
			preToken = token ;
		}
		return features ;
	}
/*	
	public Counter<String> getGramFeatures(String name) 
	{
		// TODO Auto-generated method stub
		char[] characters = name.toCharArray();
		Counter<String> features = new Counter<String>();
	
		if(characters.length == 0) //checking boundary conditions
		{
			return features ;
		}
		else if(characters.length == 1) //checking boundary conditions
		{
			char character1 = characters[0];
			features.incrementCount("UNI-" + character1, 1.0);
			return features ;
		}
		else if(characters.length == 2) //checking boundary conditions
		{
			char character1 = characters[0];
			char character2 = characters[1];
			
			features.incrementCount("UNI-" + character1, 1.0);
			features.incrementCount("UNI-" + character2, 1.0);
			features.incrementCount("BI-" + character1+character2, 1.0);
			
			return features ;
		}
		
		else if(characters.length == 3) //checking boundary conditions
		{
			char character1 = characters[0];
			char character2 = characters[1];
			char character3 = characters[2];
			
			features.incrementCount("UNI-" + character1, 1.0);
			features.incrementCount("UNI-" + character2, 1.0);
			features.incrementCount("UNI-" + character3, 1.0);
			
			features.incrementCount("BI-" + character1+character2, 1.0);
			features.incrementCount("BI-" + character2+character3, 1.0);
			
			features.incrementCount("TRI-" + character1+character2+character3, 1.0);
			
			
			return features ;
		}
		
		
		char character1 = characters[0];
		char character2 = characters[1];
		char character3 = characters[2];
		char character4 = characters[3];
		
		
		for (int i = 4; i < characters.length; i++) 
		{
			char character5 = characters[i];
			features.incrementCount("UNI-" + character1, 1.0);
			features.incrementCount("BI-" + character1+character2, 1.0);
			features.incrementCount("TRI-" + character1+character2+character3, 1.0);
			features.incrementCount("QUADI-" + character1+character2+character3+character4, 1.0);
		//	features.incrementCount("PENTA-" + character1+character2+character3+character4+character5, 1.0);
			
			
			character1 = character2 ;
			character2 = character3 ;
			character3 = character4 ;
			character4 = character5 ;
		}
		return features;
	}
*/	

	public Counter<String> getGramFeatures(String name) 
	{
		// TODO Auto-generated method stub
		char[] characters = name.toCharArray();
		Counter<String> features = new Counter<String>();
	
		if(characters.length == 0) //checking boundary conditions
		{
			return features ;
		}
		else if(characters.length == 1) //checking boundary conditions
		{
			char character1 = characters[0];
			features.incrementCount("UNI-" + character1, 1.0);
			return features ;
		}
		else if(characters.length == 2) //checking boundary conditions
		{
			char character1 = characters[0];
			char character2 = characters[1];
			
			features.incrementCount("UNI-" + character1, 1.0);
			features.incrementCount("UNI-" + character2, 1.0);
			features.incrementCount("BI-" + character1+character2, 1.0);
			
			return features ;
		}
		
		char character1 = characters[0];
		char character2 = characters[1];
		char character3 = characters[2];
		
		
		for (int i = 3; i < characters.length; i++) 
		{
			char character4 = characters[i];
			features.incrementCount("UNI-" + character1, 1.0);
			features.incrementCount("BI-" + character1+character2, 1.0);
			features.incrementCount("TRI-" + character1+character2+character3, 1.0);
			features.incrementCount("QUADI-" + character1+character2+character3+character4, 1.0);
			
			character1 = character2 ;
			character2 = character3 ;
			character3 = character4 ;
		}
		return features;
	}

	public Counter<String> getNumberFeatures(String name) 
	{
		// TODO Auto-generated method stub
		char[] characters = name.toCharArray();
		Counter<String> features = new Counter<String>();
	
		boolean numberPresent = TextUtil.checkNumber(characters);
		if (numberPresent)
		{
			features.setCount("NUM-", 1.0);
		}
		return features;
	}

	public Counter<String> getCompanyFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		
		String tokens[] = name.split("\\s++");
		
		for ( String company : companies )
		{
			if ( name.toLowerCase().contains(company))
			{
				features.setCount("COMPANY-", 1.0);
			}
		}
		return features ;
	}
	
	public Counter<String> getMovieFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		
		String tokens[] = name.split("\\s++");
		
		
		for ( String movie : movies )
		{
			for ( String token : tokens )
			{
				if ( token.toLowerCase().equalsIgnoreCase(movie))
				{
				//	System.out.println("movie name " + name );
					features.setCount("MOVIES1-", 1.0);
				}
			}
		}
		return features ;
	}
	
	
	public Counter<String> getPeopleFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		
		String tokens[] = name.split("\\s++");
		
		for ( String person : people )
		{
			for ( String token : tokens )
			{
				if ( token.toLowerCase().equalsIgnoreCase(person))
				{
					features.setCount("PERSON-", 1.0);
				}
			}
		}
		return features ;
	}
	
	
	
	public Counter<String> getPlaceFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		String tokens[] = name.split("\\s++");
		for ( String place : places )
		{
			for ( String token : tokens )
			{
				if ( token.toLowerCase().equalsIgnoreCase(place))
				{
					features.setCount("PLACE-", 1.0);
				}
			}
		}
		return features ;
	}
	
	
	
	public Counter<String> getMedicineEndFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		String tokens[] = name.split("\\s++");
		for ( String medicine : drugends )
		{
			for ( String token : tokens )
			{
				if ( token.toLowerCase().endsWith(medicine))
				{
					features.setCount("MEDICINE2-", 1.0);
				}
			}
		}
		
		return features ;
	}
	
	public Counter<String> checkRomanNumerals (String name )
	{
		Counter<String> features = new Counter<String>();
		String tokens[] = name.split("\\s++");
		for ( String token : tokens )
		{
			if (token.trim().equalsIgnoreCase("I") )
			{
				features.setCount("ROMAN-NUM", 1);
			}
			else if (token.trim().equalsIgnoreCase("II") )
			{
				features.setCount("ROMAN-NUM", 1);
			}
			else if (token.trim().equalsIgnoreCase("III") )
			{
				features.setCount("ROMAN-NUM", 1);
			}
			else if (token.trim().equalsIgnoreCase("IV") )
			{
				features.setCount("ROMAN-NUM", 1);
			}
			else if (token.trim().equalsIgnoreCase("V") )
			{
				features.setCount("ROMAN-NUM", 1);
			}
		}
		
		return features ;
	}
	
	
	public Counter<String> getMedicineStartFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		String tokens[] = name.split("\\s++");
		for ( String medicine : drugstart )
		{
			for ( String token : tokens )
			{
				if ( token.toLowerCase().startsWith(medicine))
				{
					features.setCount("MEDICINE1-", 1.0);
				}
			}
		}
		
		return features ;
	}
	
	public Counter<String> getMedicineFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		String tokens[] = name.split("\\s++");
		for ( String medicine : medicines )
		{
			for ( String token : tokens )
			{
				if ( token.toLowerCase().equalsIgnoreCase(medicine))
				{
					features.setCount("MEDICINE-", 1.0);
				}
			}
		}
		
		return features ;
	}
	
	public Counter<String> getPlacePatternFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		String tokens[] =name.split("//s++");
		
		for ( String placeE : placeEnds )
		{
			for ( String token : tokens )
			{
				if(token.toLowerCase().endsWith(placeE))
	//			if ( name.toLowerCase().endsWith(placeE))
				{
					features.setCount("PLACE_ENDS", 1.0) ;
				}
			}
		}
		return features ;
	}
	
	public Counter<String> checkStopWords(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		String tokens[] =name.split("//s++");
		
		for ( String stop : stopWords )
		{
			for ( String token : tokens )
			{
				if(token.toLowerCase().endsWith(stop))
	//			if ( name.toLowerCase().endsWith(placeE))
				{
					features.setCount("STOP_WORD", 1.0) ;
				}
			}
		}
		return features ;
	}
	
	
	
	public Counter<String> getCountryPatternFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		for ( String country : countries )
		{
		
			if(name.toLowerCase().equals(country.toLowerCase()))
			{
		//		System.out.println("country1 = "+ name);
				features.setCount("COUNTRY1-", 1) ;
			}
		}
		return features ;
	}
	
	
	
	
	public Counter<String> getPunctuationFeatures ( String name )
	{
		Counter<String> features = new Counter<String>();
		
		//check if all tokens end with period
		String tokens[] = name.split("\\s++");
	
		boolean allPeriods = false ;
		for ( String token : tokens )
		{
			char last = token.charAt(token.length()-1);
			if ( last == '.')
			{
				allPeriods = true ;
			}
			else
			{
				allPeriods = false ;
				break;
			}
		}
		if ( allPeriods)
		{
	//		features.setCount("PERIOD_END", 1.0);
			
		}
		
		//internal period
		for ( String token : tokens )
		{
			for ( int i = 0 ; i < token.length()-1 ; i++ ) 
			{
				char last = token.charAt(i);
				if ( last == '.')
				{
					features.incrementCount("PERIOD_INTERNAL", 1.0);
					
				}
			}
		}
	
		//internal ampersand - very common in many things
		for ( String token : tokens )
		{
			if ( token.equalsIgnoreCase("&"))
			{
	//			System.out.println("token & " +name);
	//			features.setCount("PERIOD_AMP", 1.0);
			}
		}
		
		for ( String token : tokens )
		{
			for ( int i = 0 ; i < token.length(); i++ )
			{
				if ( token.contains("'"))
				{
		//			features.setCount("PERIOD_APOS", 1.0);
				}
			}
		}
		
		if(name.contains("-"))
		{	
			features.setCount("PERIOD-HYPHEN", 1.0);
		}
		
		
		return features ;
	}
	
	
	public Counter<String> PatternFeatures ( String name )
	{
		Counter<String> features = new Counter<String>();
		
		char[] chars = name.toCharArray() ;
		String pattern = new String() ;
		
		int pat = 0 ; //1 = CAPS, 2 = LOWER, 3 = DIGIT, 4 = SPACE 5 = ELSE
		int old = 0 ;
		for ( int i = 0 ; i < chars.length ; i++ )
		{
			char c = chars[i];
			if ( 'A' <= c  && c <= 'Z')
			{
				pattern += "A" ;
			}
			
			else if ( 'a' <= c  && c <= 'z')
			{
				pattern += "a" ;
			}
			else if ( '0' <= c  && c <= '9')
			{
				pattern += "1" ;
			}
			else if ( c == ' ')
			{
				pattern += "S" ;
			}
		/*	
			else if ( c == '-')
			{
				pat = 5 ;
				if ( pat != old )
				{
					pattern += "S" ;
					old = 5 ;
				}
			}
		*/	
			else 
			{
				pattern += "E" ;
			
			}
			
		}
		features.setCount(pattern, 1.0) ;
		return features;
	
	}
	
	public Counter<String> getSummarizedPatternFeatures ( String name )
	{
		Counter<String> features = new Counter<String>();
		
		char[] chars = name.toCharArray() ;
		String pattern = new String() ;
		
		String pattern2 = new String() ;
		
		int pat = 0 ; //1 = CAPS, 2 = LOWER, 3 = DIGIT, 4 = SPACE 5 = ELSE
		int old = 0 ;
		for ( int i = 0 ; i < chars.length ; i++ )
		{
			char c = chars[i];
			if ( 'A' <= c  && c <= 'Z')
			{
				pat = 1 ;
				if ( pat != old )
				{
					pattern += "A" ;
					pattern2 += 'A' ;
					old = 1 ;
				}
			}
			else if ( 'a' <= c  && c <= 'z')
			{
				pat = 2 ;
				if ( pat != old )
				{
					pattern += "a" ;
					pattern2 += 'a' ;
					
					old = 2 ;
				}
			}
			else if ( '0' <= c  && c <= '9')
			{
				pat = 3 ;
				if ( pat != old )
				{
					pattern += "1" ;
			//		pattern2 += '1' ;
					
					old = 3 ;
				}
			}
			else if ( c == ' ')
			{
				pat = 4 ;
				if ( pat != old )
				{
					pattern += "S" ;
		//			pattern2 += 'S' ;
					
					old = 4 ;
				}
			}
	/*		
			else if ( c == '-')
			{
				pat = 5 ;
				if ( pat != old )
				{
					pattern += "D" ;
					old = 5 ;
				}
			}
	*/		
			else 
			{
				pat = 6 ;
				if ( pat != old )
				{
					pattern += "E" ;
					old = 6 ;
				}
			}
		}
	//	System.out.println(name);
		features.setCount(pattern, 1.0) ;
		
	//	features.setCount(pattern2, 1.0) ;
		
		return features;
		
	}
	

	public Counter<String> getCAPSFeatures(String name) 
	{
		// TODO Auto-generated method stub
		Counter<String> features = new Counter<String>();
		
		
		String tokens[] = name.split("\\s++") ;
		
		for ( String token : tokens )
		{
			boolean allUpperCase = false ;
			
			
			for ( int i = 0 ; i < token.length() ; i++ )
			{
				char c = token.charAt(i);
				if ( 'A' <= c && c <= 'Z')
				{
					allUpperCase = true ;
				}
				else
				{
					allUpperCase = false ;
					break;
				}
				if(i == 0 )
				{
					features.setCount("CAPS_BEGIN", 1.0);
				}
				
			}
			if(allUpperCase)
			{
				features.setCount("CAPS_ALL", 1.0);
			}
		}
		
		return features ;
	}
	
	
	
	
}
