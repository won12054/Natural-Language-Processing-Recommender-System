import json
import pandas as pd
from apyori import apriori
'''
1.	Load the data and carry out some basic data analysis and exploration. 
Note the results of your analysis in your analysis report. 
At minimum, carry out the following:
  a. Note the total number of instances of recipes
  b. Note the number of cuisines available in the data.
  c. Create a table illustrating each cuisine type and number of recipes available in the file related to that cuisine.
'''
file_path = 'C:/Users/Public/6th/NLP and Recommender Systems/assignment3/recipies.json'
with open(file_path, 'r', encoding='utf-8') as file:
    recipes_data = json.load(file)
df = pd.DataFrame(recipes_data)

df.head()

# unhashable type 'list'
df['ingredients'] = df['ingredients'].apply(lambda x: ', '.join(x))

df.duplicated().sum()

df.isnull().sum()

df = pd.DataFrame(recipes_data)

df.head()

total_recipes = len(df)
print(f"Total number of recipes: {total_recipes}")

unique_cuisines = df['cuisine'].nunique()
print(f"Number of unique cuisines: {unique_cuisines}")

cuisine_counts = df['cuisine'].value_counts().reset_index()
cuisine_counts.columns = ['Cuisine', 'Number of Recipes']
print("Cuisine and Recipe Counts:")
print(cuisine_counts)

'''
2. Your app should receive a 'cuisine type' as input from the user, 
for example, 'Greek', 'Italian'…etc. 
If the 'cuisine type' is not available, 
then reply to the user "We don't have recommendations for XXX"
where XXX is the inputted cuisine type. 
Then prompt the user to enter a different 'cuisine type'. 
(hint: use python input())
'''
def get_cuisine_recommendations():
    available_cuisines = df['cuisine'].unique()
    
    while True:
        user_input = input("\nEnter a cuisine type: ").strip()
        
        if user_input in available_cuisines:
            recommendations = df[df['cuisine'] == user_input]
            print(f"\nHere are some recipes for {user_input} cuisine:")
            print(recommendations[['cuisine', 'ingredients']].head())  
            break
        else:
            print(f"We don’t have recommendations for {user_input}.")

get_cuisine_recommendations()

'''
3. If the 'cuisine type' recipe data is available in the json file, then:
  a. Analyze all the ingredients available under the inputted "cuisine type" using the apriori algorithm, 
     according to the following parameters:
    i. Set the support value to  100/total # of recipes for the selected cuisine 
    ii. Set the confidence value to 0.5
'''
def analyze_cuisine_ingredients_with_apyori(df, cuisine_type):
    cuisine_data = df[df['cuisine'] == cuisine_type]
    
    ingredient_list = cuisine_data['ingredients'].tolist()
    
    total_recipes = len(cuisine_data)
    support_threshold = 100 / total_recipes  

    rules = apriori(
        transactions=ingredient_list,
        min_support=support_threshold,
        min_confidence=0.5
    )

    results = list(rules)
    
    return results

'''
4. Present back to the user the following:
  a. The top group of ingredients that the algorithm calculates for the inputted cuisine type, 
     i.e. the most frequent dataset. 
     (hint: This would be stored in the first record of the RelationRecords returned from the algorithm)
  b. All rules with lift value greater than two.  
'''
def present_results(results):
    if not results:
        print("No frequent itemsets or rules found.")
        return
        
    # a: Top group of ingredients
    sorted_results = sorted(results, key=lambda x: x.support, reverse=True)
    top_group = sorted_results[0]
    
    print("\nTop Group of Ingredients:")
    print(f"Items: {top_group.items}")
    print(f"Support: {top_group.support}")

    # b: Rules with Lift > 2
    print("\nRules with Lift > 2:")
    found = False
    for rule in results:
        for ordered_stat in rule.ordered_statistics:
            if ordered_stat.lift > 2:
                found = True
                print(f"Rule: {list(ordered_stat.items_base)} -> {list(ordered_stat.items_add)}")
                print(f"  Support: {rule.support}")
                print(f"  Confidence: {ordered_stat.confidence}")
                print(f"  Lift: {ordered_stat.lift}")
                print("-" * 40)
    if not found:
        print("No rules with lift > 2 were found.")

'''
5. Continue accepting input from the user and responding until the user enters an “exit” text.
'''
def get_cuisine_recommendations_with_analysis():
    available_cuisines = df['cuisine'].unique()  
    
    while True:
        user_input = input("\nEnter a cuisine type (or type 'exit' to quit): ").strip().lower()
        
        if user_input == "exit":
            print("Exiting the program. Thank you!")
            break
        
        if user_input in available_cuisines:
            print(f"\nAnalyzing recipes for {user_input} cuisine...")
            results = analyze_cuisine_ingredients_with_apyori(df, user_input)
            present_results(results)
        else:
            print(f"We don’t have recommendations for {user_input}. Please try again.")

get_cuisine_recommendations_with_analysis()
