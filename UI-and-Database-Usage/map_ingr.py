#import the pandas library
import json
from math import ceil
import pandas as pd

#unpickle the ingredient map
ingr_map = pd.read_pickle('ingr_map.pkl')

#get dataframes for all the needed csv files:
recipe_df = pd.read_csv('PP_recipes.csv')
raw_recipes_df = pd.read_csv('RAW_recipes.csv')
raw_ratings_df = pd.read_csv('RAW_interactions.csv')


def get_ingr_ids(list_of_ingr):
    """map list of ingredients to return a list of ingr ids
            Parameter: list_of_ingr (a list of ingredient NAMES)
            Returns a list of ingredient IDs"""
    LoIngrIndex = []
    allIngr = ingr_map['replaced'].tolist()
    allFullNameIngr = ingr_map['raw_ingr'].tolist()
    for ingr in list_of_ingr:
        if ingr in set(allIngr):
            ingrRow = allIngr.index(ingr)
            ingr_id = ingr_map.iloc[ingrRow]['id']
            LoIngrIndex.append(ingr_id) #add to list of ingr ids
        else:
            if ingr in set(allFullNameIngr):
                ingrRow = allFullNameIngr.index(ingr)
                ingr_id = ingr_map.iloc[ingrRow]['id']
            else:
                print(f"Uh oh, Ingr: {ingr} not recognized")
    return LoIngrIndex


def map_recipe_id_name(recipe_id):
    """Given a recipe id (int)
    Returns a recipe name (str)"""
    # turn all the recipe ids into a list
    all_recipe_ids = raw_recipes_df['id'].tolist()
    # get the index of the recipe ID
    recipe_ind = all_recipe_ids.index(recipe_id)
    # get the name of the recipe from the row in the df that corresponds to the index
    return raw_recipes_df.iloc[recipe_ind]['name']



def get_avg_rating(recipe_id):
    """given a recipe_id, pull all of the relevant ratings, and calculate total and avg ratings for the recipe
        Returns the total number of ratings, avg rating"""
    #generate dataframe for all the reviews that have a recipe_id matching recipe_id
    relevant_ratings_df = raw_ratings_df.query('recipe_id == @recipe_id')

    #get list of the rating from the relevant reviews:
    all_ratings = relevant_ratings_df['rating'].tolist()
    num_ratings = len(all_ratings)
    avg_rating = sum(all_ratings)/num_ratings
    return num_ratings, avg_rating

def order_ratings(recipe_pool):
    """Given a list of recipe IDs in our pool, 
        return an ordered list of recipe ids (highest -> lowest weighted rating)"""
    recipe_pool_dict = {}
    for id in recipe_pool:
        total_r, avg_r = get_avg_rating(id)
        recipe_pool_dict[id] = total_r*avg_r

    recipe_pool_ordered = {key: val for key, val in sorted(recipe_pool_dict.items(), key=lambda item: item[1], reverse=True)}

    #We may want to add something here to only return part of the ordered pool of recipes!
    return recipe_pool_ordered

def display_recipe(recipe_pool):
    """Given recipe_pool (a list of recipe IDs in our final pool)
    Display/Print all recipes for the user"""
    all_recipe_ids = raw_recipes_df['id'].tolist()
    recipe_counter = 1
    while recipe_counter <= 5 and recipe_counter<= len(recipe_pool):
        recipe_id = recipe_pool[recipe_counter-1]
        recipe_title = map_recipe_id_name(recipe_id)
        num_rating, avg_rating = get_avg_rating(recipe_id)
        rating_str = f'Rating: {avg_rating:2f} among {num_rating} users'
        recipe_ind = all_recipe_ids.index(recipe_id)
        recipe_row = raw_recipes_df.iloc[recipe_ind]
        time_est = f'{recipe_row["minutes"]} minutes'
        ingredients = recipe_row["ingredients"]
        description = recipe_row["description"]
        instruction = eval(recipe_row['steps'])

        print(f'Recipe #{recipe_counter}')
        #print recipe title
        print(recipe_title)
        #print ratings, number of ratings
        print(rating_str)
        # print time estimate
        print(time_est)
        #print ingredients
        print('Ingredients:', ingredients)
        #print instructions
        print("Steps:")
        for i in range(len(instruction)):
            print(i, instruction[i])
        print()
        recipe_counter += 1


def get_ingredients(recipe_id):
    """given a recipe id, get its ingredients
    Return a list of ingredient IDs, for recipe with given ID"""
    all_recipe_ids = recipe_df['id'].tolist()
    recipe_ind = all_recipe_ids.index(recipe_id)
    recipe_row = recipe_df.iloc[recipe_ind]
    ingredients = recipe_row["ingredient_ids"]
    #print(ingredients)
    return eval(ingredients)



def get_num_missing_ingredients(recipe_id, list_of_ingr):
    """given a recipe id and a list of ingredients, return number of missing ingredients
    Parameters: recipe_id (int id of recipe)
    list_of_ingr: list of ingredient IDs  
    Returns number of ingredients missing from recipe's ingredients"""

    recipe_ingredients = get_ingredients(recipe_id)
    r_main_ingr = get_main_ingr(recipe_id)
    #recipe_ingredients = [4308, 1910, 1168, 1, 2]
    counter = 0
    for id in recipe_ingredients:
        if id not in list_of_ingr:
            if id in r_main_ingr:
                counter += 1
            else:
                counter += 0.25
    return counter 

main_ingredient_map = pd.read_csv('map_main_ingredients.csv')

def get_main_ingr(recipe_id):
    all_recipes = main_ingredient_map['Recipe IDs'].tolist()
    all_ingr = main_ingredient_map['Main Ingredient IDs'].tolist()
    ingredientL = []
    for i in range(len(all_recipes)):
        if recipe_id in eval(all_recipes[i]):
            ingredientL.append(all_ingr[i])
    #print(ingredientL)
    return ingredientL



#have not tested this much 

def find_candidates(list_of_ingr):
    """given a list of ingredients, find recipes to consider recommending
    list_of_ingr: list of ingredient IDs  
    Returns list of dictionaries of recipe ID + number of missing ingredients"""
    candidates = {}
    #treat each ingredient as main ingredient
    all_ingr_ids = main_ingredient_map['Main Ingredient IDs'].tolist()
    for ingr in list_of_ingr: 
        if ingr in all_ingr_ids:
            ingr_ind = all_ingr_ids.index(ingr)

            ingr_recipes = eval(main_ingredient_map.iloc[ingr_ind]["Recipe IDs"])

            for recipe in ingr_recipes:

                #main_ingr_count = 0
                #r_main_ingr = get_main_ingr(map_recipe_id_name(recipe))
                # for i in r_main_ingr:
                #     if i in set(list_of_ingr):
                #         main_ingr_count += 1
                # if main_ingr_count == len(r_main_ingr):
                #     print(172)
                recipe_ingredients = get_ingredients(recipe)
                
                #recipe_ingredients = [4308, 1910, 1168, 1, 2]

                #maybe there's a better way to do this
                
                missing_ingredients = get_num_missing_ingredients(recipe, list_of_ingr)
                portion_missing = missing_ingredients/len(recipe_ingredients)
                candidates[recipe] = portion_missing * 100 #duplicate recipes won't appear twice in the dictionary 
                    
    #print("checkpoint!")

    #filter out too many missing ingr:
    candidates = {key: val for key, val in sorted(candidates.items(), key=lambda item: item[1])}
    print(candidates)
    recipe_candidates = list(candidates.keys())
    twenty_perc_mark = int(len(recipe_candidates)*0.2)
    short_list = recipe_candidates[:twenty_perc_mark]

    #order by ratings:
    short_list = order_ratings(short_list)
    ordered_candidates = {}
    for recipe in short_list:
        ordered_candidates[recipe] = candidates[recipe]
    return ordered_candidates

    
ingredients = ["cream cheese", "chicken", "lettuce", "eggs", "milk", "butter", "bacon", "fresh chive", "white vinegar", "cheddar", "sour cream", "paprika"]
#ingredients_ids = get_ingr_ids(ingredients)
#print(find_candidates(ingredients_ids))