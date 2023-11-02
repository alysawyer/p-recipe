#import the pandas library

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
    """Given a dictionary of recipe IDs in our pool, 
        return an ordered list of recipe ids (highest -> lowest weighted rating)"""
    recipe_pool_rate_dict = {}
    for id in recipe_pool:
        total_r, avg_r = get_avg_rating(id)
        recipe_pool_rate_dict[id] = total_r*avg_r

    #get a recipe_pool dictionary with recipe id: score NOT rating

    #recipe_pool dictionary {id:rating} sorted
    recipe_pool_rate_dict = {key: value for key, value in sorted(recipe_pool_rate_dict.items(), key=lambda item: item[1], reverse=True)}

    # recipe_pool sorted by rating - just ids:
    recipe_rating_sorted_keys = list(recipe_pool_rate_dict.keys())
    recipe_pool_D = {id: recipe_pool[id] for id in recipe_rating_sorted_keys}
    #We may want to add something here to only return part of the ordered pool of recipes!
    return recipe_pool_D

def limit_time(recipe_pool, max_minutes):
    """Given a dictionary of recipe IDs in our pool and a time limit in mimutes,
        return  the pool without recipes that exceed the max time"""
    all_recipe_ids = raw_recipes_df['id'].tolist()

    filtered_recipes = {}

    for id, score in recipe_pool.items():
        recipe_ind = all_recipe_ids.index(id)
        time = raw_recipes_df.iloc[recipe_ind]['minutes']
        if(time <= max_minutes):
            filtered_recipes[id] = score

    return filtered_recipes

def order_times(recipe_pool):
    """Given a dictionary of recipe IDs in our pool, 
        return an ordered list of recipe ids (lowest -> highest time to make)"""
    recipe_pool_times = {}
    all_recipe_ids = raw_recipes_df['id'].tolist()

    for id in recipe_pool:
        recipe_ind = all_recipe_ids.index(id)
        recipe_pool_times[id] = raw_recipes_df.iloc[recipe_ind]['minutes']
    recipe_pool_times = {key: value for key, value in sorted(recipe_pool_times.items(), key=lambda item: item[1], reverse=False)}
    
    sorted_keys = list(recipe_pool_times.keys())
    recipe_pool_D = {id: recipe_pool[id] for id in sorted_keys}
    return recipe_pool_D

def order_times(recipe_pool):
    """Given a dictionary of recipe IDs in our pool, 
        return an ordered list of recipe ids (lowest -> highest time to make)"""
    recipe_pool_times = {}
    all_recipe_ids = raw_recipes_df['id'].tolist()

    for id in recipe_pool:
        recipe_ind = all_recipe_ids.index(id)
        recipe_pool_times[id] = raw_recipes_df.iloc[recipe_ind]['minutes']
    recipe_pool_times = {key: value for key, value in sorted(recipe_pool_times.items(), key=lambda item: item[1], reverse=False)}
    
    sorted_keys = list(recipe_pool_times.keys())
    recipe_pool_D = {id: recipe_pool[id] for id in sorted_keys}
    return recipe_pool_D

def order_fats(recipe_pool):
    """Given a dictionary of recipe IDs in our pool, 
        return an ordered list of recipe ids (lowest -> highest total fat)"""
    recipe_pool_fats = {}
    all_recipe_ids = raw_recipes_df['id'].tolist()

    for id in recipe_pool:
        recipe_ind = all_recipe_ids.index(id)
        recipe_nutrition = eval(raw_recipes_df.iloc[recipe_ind]['nutrition'])
        recipe_pool_fats[id] = recipe_nutrition[1]
        #print(map_recipe_id_name(id), " ", recipe_nutrition[1])
    recipe_pool_fats = {key: value for key, value in sorted(recipe_pool_fats.items(), key=lambda item: item[1], reverse=False)}
    
    sorted_keys = list(recipe_pool_fats.keys())
    recipe_pool_D = {id: recipe_pool[id] for id in sorted_keys}
    return recipe_pool_D

def order_sugars(recipe_pool):
    """Given a dictionary of recipe IDs in our pool, 
        return an ordered list of recipe ids (lowest -> highest total amount of sugar)"""
    recipe_pool_sugars = {}
    all_recipe_ids = raw_recipes_df['id'].tolist()

    for id in recipe_pool:
        recipe_ind = all_recipe_ids.index(id)
        recipe_nutrition = eval(raw_recipes_df.iloc[recipe_ind]['nutrition'])
        recipe_pool_sugars[id] = recipe_nutrition[2]
    recipe_pool_sugars = {key: value for key, value in sorted(recipe_pool_sugars.items(), key=lambda item: item[1], reverse=False)}
    
    sorted_keys = list(recipe_pool_sugars.keys())
    recipe_pool_D = {id: recipe_pool[id] for id in sorted_keys}
    return recipe_pool_D



def display_recipe(recipe_pool,subs_needed):
    """Given recipe_pool (a dictionary of recipe IDs:score in our final pool)
    Display/Print 5 recipes for the user"""
    all_recipe_ids = raw_recipes_df['id'].tolist()
    recipe_ids_pool = list(recipe_pool.keys())
    recipe_counter = 1
    while recipe_counter <= 5 and recipe_counter<= len(recipe_ids_pool):
        recipe_id = recipe_ids_pool[recipe_counter-1]
        num_missing = recipe_pool[recipe_id]
        recipe_title = map_recipe_id_name(recipe_id)
        num_rating, avg_rating = get_avg_rating(recipe_id)
        rating_str = f'Rating: {avg_rating:2f} among {num_rating} users'
        recipe_ind = all_recipe_ids.index(recipe_id)
        recipe_row = raw_recipes_df.iloc[recipe_ind]
        time_est = f'{recipe_row["minutes"]} minutes'
        ingredients = recipe_row["ingredients"]
        description = recipe_row["description"]
        instruction = eval(recipe_row['steps'])
        subs_used = subs_needed[recipe_id]
        using_subs = True
        if subs_needed[recipe_id] == {}:
            using_subs = False

        print(f'Recipe #{recipe_counter}')
        print()
        #print recipe title
        print(recipe_title)
        print(f"MISSING {num_missing} INGREDIENTS")
        print(f"Using SUBSTITUES: {using_subs}")
        for key, value in subs_used.items():
            print(f"- Use {value} instead of {key}")
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
    #recipe_ingredients = [4308, 1910, 1168, 1, 2]
    counter = 0
    for id in recipe_ingredients:
        if id not in list_of_ingr:
            counter += 1
    return counter 

recipe_map = pd.read_csv('recipe_main_ingr_map.csv')
sub_map = pd.read_csv('ingr_subs.csv')
# subs_needed_dict tracks subs needed for each recipe:
        # {recipe id: {ingr: sub}}

def get_sub(l_ingr_ids, missing_ingr):
    """given a list of ingr ids and a missing ingr id, return a working sub if one exists"""
    all_subs = sub_map['ingr_id'].tolist() #a list of all ingr that have subs
    if missing_ingr in all_subs:
        sub_index = all_subs.index(missing_ingr)
        sub_opts = eval(sub_map.iloc[sub_index]['substitutions'])
        for sub_set in sub_opts:
            count_missing = 0
            for ingr in sub_set:
                if ingr not in l_ingr_ids:
                    count_missing += 1
            if count_missing == 0:
                return (missing_ingr, sub_set)
    return (missing_ingr, []) #we couldn't find a working sub

def working_sub_exists(l_ingr_ids, missing_ingr):
    """given a list of ingr ids and a missing ingredient's id, determine if a substitute is available, and if we can use it with our ingredients
    If we can use a sub: return (True, missing ingr, sub); otherwise (False, missing ingr, [])"""
    all_subs = sub_map['ingr_id'].tolist() #a list of all ingr that have subs
    if missing_ingr in all_subs:
        sub_index = all_subs.index(missing_ingr)
        sub_opts = eval(sub_map.iloc[sub_index]['substitutions'])
        for sub_set in sub_opts:
            count_missing = 0
            for ingr in sub_set:
                if ingr not in l_ingr_ids:
                    count_missing += 1
            if count_missing == 0:
                return (True, missing_ingr, sub_set)
    return (False, missing_ingr, []) #we couldn't find a working sub




def get_recipe_pool(l_ingr_ids):
    subs_needed_dict = {}
    recipe_pool = {}
    
    # {recipe id: score}
    #score = #produce missing + # dairy missing + # other missing
    # only allow to be added to pool if zero protein missing, <=2 produce missing, <= 2 dairy missing, <= 6 other missing

    #iterate through every recipe
    for recipe_ind in recipe_map.index:
        recipe_ingr = eval(recipe_map.iloc[recipe_ind]['ingr_ids'])
        recipe_protein = recipe_ingr[0]
        recipe_produce = recipe_ingr[1]
        recipe_dairy = recipe_ingr[2]
        recipe_other = recipe_ingr[3]
        recipe_id = recipe_map.iloc[recipe_ind]['recipe_id']

        #track which subs we will need to use if recipe stays:
        subs_for_recipe = {}

        #protein:
        should_include = True
        for prot in recipe_protein:
            if prot not in l_ingr_ids:
                subs = working_sub_exists(l_ingr_ids, prot)
                if not subs[0]:
                    should_include = False
                    break
                else:
                    subs_for_recipe[prot] = subs[2]
                
                    
        if should_include:
            num_prod_miss = 0
            should_inc_prod = True
            for prod in recipe_produce:
                if prod not in l_ingr_ids:
                    subs = working_sub_exists(l_ingr_ids, prod)
                    if not subs[0]:
                        num_prod_miss += 1
                        if num_prod_miss > 1:
                            should_inc_prod = False
                            break
                    else:
                        subs_for_recipe[prod] = subs[2]
            if should_inc_prod:
                num_dairy_miss = 0
                should_inc_dairy = True
                for dairy in recipe_dairy:
                    if dairy not in l_ingr_ids:
                        subs = working_sub_exists(l_ingr_ids, dairy)
                        if not subs[0]:
                            num_dairy_miss += 1
                            if num_dairy_miss > 1:
                                should_inc_dairy = False
                                break
                        else:
                            subs_for_recipe[dairy] = subs[2]
                if should_inc_dairy:
                    num_other_miss = 0
                    should_inc_o = True
                    for other in recipe_other:
                        if other not in l_ingr_ids:
                            subs = working_sub_exists(l_ingr_ids, other)
                            if not subs[0]:
                                num_other_miss += 1
                                if num_other_miss > 5:
                                    should_inc_o = False
                                    break
                            else:
                                subs_for_recipe[other] = subs[2]
                    if should_inc_o:
                        score = num_other_miss + num_dairy_miss + num_prod_miss
                        recipe_pool[recipe_id] = score
                        subs_needed_dict[recipe_id] = subs_for_recipe
    return recipe_pool, subs_needed_dict #return the reecipe pool, and the subs that would be used for the recipes

def filter_pool(recipe_pool):
    #score goes from 0-10
    recipe_scores = sorted(list(recipe_pool.values()))
    best_score = min(recipe_scores)
    worst_score = max(recipe_scores)
    total_scores = sum(recipe_scores)
    num_scores = len(recipe_scores)

    filtered_recipes = {}
    score_marker = 10
    #score_marker is the max score a recipe can have to be included

    #if a score <= 1 --> only report these scores
    if best_score <= 1:
        score_marker = 1
    elif num_scores <= 5:
        score_marker = 10
    elif num_scores <= 10:
        half_len = num_scores//2
        score_marker = recipe_scores[half_len]
    elif num_scores <= 25:
        quarter_len = num_scores//4
        score_marker = recipe_scores[quarter_len]
    else:
        score_marker = recipe_scores[25]

    for recipe, score in recipe_pool.items():
        if score <= score_marker:
            filtered_recipes[recipe] = score
    
    return {key: val for key, val in sorted(filtered_recipes.items(), key=lambda item: item[1], reverse=True)}

def main(ingredients):
    ingr_ids = get_ingr_ids(ingredients)
    recipe_pool, subs_needed = get_recipe_pool(ingr_ids)
    filtered_pool = filter_pool(recipe_pool)
    ordered_by_rating_pool = order_ratings(filtered_pool)
    display_recipe(ordered_by_rating_pool, subs_needed)


    
ingredients = ["cream cheese", "chicken", "lettuce", "eggs", "milk", "butter", "bacon", "fresh chive", "white vinegar", "cheddar", "sour cream", "paprika"]
main(ingredients)