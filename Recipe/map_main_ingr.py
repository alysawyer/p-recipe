# create a csv data file with:
# a column for main ingredient (contains all ingredients): "Main Ingredient IDs" 
# a column for corresponding recipes: "Recipe IDs"
# -1 maps to all recipes without a main ingredient 

import pandas as pd

from map_ingr import get_main_ingr, map_recipe_id_name 

ingr_map = pd.read_pickle('ingr_map.pkl')
raw_recipes_df = pd.read_csv('RAW_recipes.csv')


#recipes sorted by main ingredient
#key: main ingredient ID, value: list of recipe IDs with that main ingredient
#a recipe can appear multiple times 
sorted_recipes = {} 

for id in ingr_map['id']:
    sorted_recipes[id] = []
sorted_recipes[-1] = [] #if a recipe has no main ingredient, add here
for id in raw_recipes_df['id']:
    recipe_name = map_recipe_id_name(id)
    print(recipe_name)
    main_ingredients = get_main_ingr(str(recipe_name))
    for ingr in main_ingredients:
        sorted_recipes[ingr].append(id)
    if not main_ingredients:
        sorted_recipes[-1].append(id)

map = pd.DataFrame({'Main Ingredient IDs': sorted_recipes.keys(), 'Recipe IDs': sorted_recipes.values()})
map.to_csv("map_main_ingredients.csv", index = False)