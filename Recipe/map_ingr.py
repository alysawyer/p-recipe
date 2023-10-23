#import the pandas library
import pandas as pd

#unpickle the ingredient map
ingr_map = pd.read_pickle('ingr_map.pkl')

def get_df_from_csv(path):
    """Make a dataframe from a csv file"""
    new_df = pd.read_csv(path)
    return new_df

#get dataframes for all the needed csv files:
path_start = ''
recipe_df = get_df_from_csv('PP_recipes.csv')
raw_recipes_df = get_df_from_csv('RAW_recipes.csv')


def get_ingr_ids(list_of_ingr):
    """map list of ingredients to return a list of ingr ids
            Parameter: list_of_ingr (a list of ingredient NAMES)
            Returns a list of ingredient IDs"""
    LoIngrIndex = []
    for ingr in list_of_ingr:
        #generate dataframe for all the ingredients that have a category matching the ingr
        ingr_cat_df = ingr_map.query('replaced == @ingr')
        #get the "id" from the first row in the df (all the same id)
        ingr_id = ingr_cat_df.iloc[0]['id']
        LoIngrIndex.append(ingr_id) #add to list of ingr ids
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


def get_main_ingr(recipe_title):
    """ get ingredients in a recipe title and return a list of ingredient ids in a recipe title
            in theory identifying most significant ingredients in a recipe
            Parameters: recipe_title (a string recipe name)
            Returns a list of ingredient ids for ingredients in that title
    """
    recipe_word_l = recipe_title.split(' ')
    ingr_name_list = []
    all_ingredients = ingr_map['replaced'].tolist()

    for word in recipe_word_l:
        if word in all_ingredients:
            ingr_name_list.append(word)
    #print(ingr_name_list, "in title of recipe")
    return get_ingr_ids(ingr_name_list)

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

    recipe_pool_ordered = sorted(recipe_pool_dict, key=recipe_pool_dict.get, reverse=True)

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
    all_recipe_ids = raw_recipes_df['id'].tolist()
    recipe_ind = all_recipe_ids.index(recipe_id)
    recipe_row = raw_recipes_df.iloc[recipe_ind]
    ingredients = recipe_row["ingredients"]
    print(ingredients)
    return get_ingr_ids(ingredients)


def get_missing_ingredients(recipe_id, list_of_ingr):
    """given a recipe id and a list of ingredients, return number of missing ingredients
    Parameters: recipe_id (int id of recipe)
    list_of_ingr: list of ingredient IDs  
    Returns number of ingredients missing from recipe's ingredients"""

    recipe_ingredients = get_ingredients(recipe_id)
    counter = 0
    for id in recipe_ingredients:
        if id not in list_of_ingr:
            counter += 1
    return counter 

main_ingredients = {}
for id in raw_recipes_df['id']:
    main_ingredients[id] = get_main_ingr(str(map_recipe_id_name(id)))

print(main_ingredients[112140])
