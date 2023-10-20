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
    print(ingr_name_list, "in title of recipe")
    return get_ingr_ids(ingr_name_list)