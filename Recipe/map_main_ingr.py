# create a csv data file with:
# a column for main ingredient (contains all ingredients): "Main Ingredient IDs" 
# a column for corresponding recipes: "Recipe IDs"
# -1 maps to all recipes without a main ingredient 

import pandas as pd

from map_ingr import * 

ingr_map = pd.read_pickle('ingr_map.pkl')
recipes_df = pd.read_csv('PP_recipes.csv')

main_ingredients = [
    "chicken", "beef", "pork", "lamb", "fish", "seafood",
    "tofu", "tempeh",
    "olive oil",
    "onions",
    "garlic",
    "butter",
    "eggs",
    "milk",
    "flour",
    "sugar",
    "tomatoes",
    "rice",
    "pasta",
    "potatoes",
    "carrots",
    "bell peppers",
    "broccoli",
    "spinach",
    "lemon",
    "lime",
    "vinegar",
    "soy sauce",
    "bread",
    "cheese",
    "honey",
    "mustard",
    "cumin",
    "paprika",
    "cinnamon",
    "nutmeg",
    "baking powder",
    "vanilla extract",
    "cilantro",
    "basil",
    "mushrooms",
    "green beans",
    "corn",
    "celery",
    "ground beef",
    "chicken broth",
    "yogurt",
    "parsley",
    "thyme",
    "rosemary",
    "bay leaves",
    "oregano",
    "cayenne pepper",
    "quinoa",
    "asparagus",
    "zucchini",
    "peas",
    "curry powder",
    "onion powder",
    "garlic powder",
    "ketchup",
    "mayonnaise",
    "chocolate chips",
    "cocoa powder",
    "brown sugar",
    "powdered sugar",
    "all-purpose flour",
    "unsweetened cocoa powder",
    "vegetable oil",
    "shortening",
    "cream of tartar",
    "yeast",
    "maple syrup",
    "corn syrup",
    "molasses",
    "lemon juice",
    "orange zest",
    "almond extract",
    "cornstarch",
    "confectioners' sugar",
    "cream cheese",
    "heavy cream",
    "buttermilk",
    "sour cream",
    "baking chocolate",
    "white chocolate",
    "semisweet chocolate",
    "cocoa butter",
    "cornflour",
    "raisins",
    "nuts",
    "almonds",
    "walnuts",
    "pecans",
    "coconut",
    "oats",
    "ground ginger",
    "poppy seeds"
]

main_ingr_ids = get_ingr_ids(main_ingredients)
sorted_recipes = {}
for ingr in main_ingr_ids:
    sorted_recipes[ingr] = []
for recipe_i in recipes_df.index:
    list_of_i = eval(recipes_df.iloc[recipe_i]['ingredient_ids'])
    for ingr in main_ingr_ids:
        if ingr in list_of_i:
            sorted_recipes[ingr].append(recipes_df.iloc[recipe_i]['id'])



#recipes sorted by main ingredient
#key: main ingredient ID, value: list of recipe IDs with that main ingredient
#a recipe can appear multiple times 
# sorted_recipes = {} 

# for id in ingr_map['id']:
#     sorted_recipes[id] = []
# sorted_recipes[-1] = [] #if a recipe has no main ingredient, add here
# for id in raw_recipes_df['id']:
#     recipe_name = map_recipe_id_name(id)
#     print(recipe_name)
#     main_ingredients = get_main_ingr(str(recipe_name))
#     for ingr in main_ingredients:
#         sorted_recipes[ingr].append(id)
#     if not main_ingredients:
#         sorted_recipes[-1].append(id)

map = pd.DataFrame({'Main Ingredient IDs': sorted_recipes.keys(), 'Recipe IDs': sorted_recipes.values()})
map.to_csv("map_main_ingredients.csv", index = False)