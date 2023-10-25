import pandas as pd

from map_ingr import * 

ingr_map = pd.read_pickle('ingr_map.pkl')
recipes_df = pd.read_csv('PP_recipes.csv')

common_proteins = [
    "chicken",
    "beef",
    "pork",
    "fish",
    "salmon",
    "tuna",
    "shrimp",
    "lamb",
    "turkey",
    "duck",
    "veal",
    "venison",
    "tofu",
    "tempeh",
    "eggs",
    "quail",
    "rabbit",
    "pheasant",
    "bison",
    "goat",
    "sausage",
    "sausages",
    "seafood",
    "crab",
    "lobster",
    "scallops",
    "mussels",
    "clams",
    "oysters",
    "octopus",
    "squid",
    "cuttlefish",
    "snails",
    "anchovies",
    "mackerel",
    "trout",
    "catfish",
    "tilapia",
    "sardines",
    "haddock",
    "hake",
    "perch",
    "pangasius",
    "mahi mahi",
    "swordfish",
    "surimi",
    "quorn",
    "buffalo",
    "elk",
    "wild boar",
    "alligator",
    "kangaroo",
    "grouse",
    "guinea fowl",
]

common_produce = [
    "onion",
    "garlic",
    "potato",
    "carrot",
    "bell pepper",
    "tomato",
    "lettuce",
    "cucumber",
    "zucchini",
    "mushroom",
    "spinach",
    "kale",
    "broccoli",
    "cauliflower",
    "cabbage",
    "asparagus",
    "green beans",
    "peas",
    "corn",
    "eggplant",
    "celery",
    "avocado",
    "radish",
    "turnip",
    "sweet potato",
    "pumpkin",
    "butternut squash",
    "acorn squash",
    "cantaloupe",
    "watermelon",
    "strawberries",
    "blueberries",
    "raspberries",
    "blackberries",
    "apple",
    "pear",
    "banana",
    "grapes",
    "orange",
    "lemon",
    "lime",
    "kiwi",
    "mango",
    "pineapple",
    "pomegranate",
]

common_dairy = [
    "milk",
    "butter",
    "cheese",
    "yogurt",
    "cream",
    "sour cream",
    "whipped cream",
    "cottage cheese",
    "cream cheese",
    "evaporated milk",
    "condensed milk",
    "buttermilk",
    "half-and-half",
    "heavy cream",
    "sour milk",
    "kefir",
    "ghee",
    "mascarpone",
    "ricotta",
    "cheddar",
    "mozzarella"
]

protein_ids = set(get_ingr_ids(common_proteins))
produce_ids = set(get_ingr_ids(common_produce))
dairy_ids = set(get_ingr_ids(common_dairy))

sorted_protein_recipes = {}
sorted_produce_recipes = {}
sorted_dairy_recipes = {}

for ingr in protein_ids:
    sorted_protein_recipes[ingr] = []
for ingr in produce_ids:
   sorted_produce_recipes[ingr] = [] 
for ingr in dairy_ids:
    sorted_dairy_recipes[ingr] = []


recipe_map = {}

all_recipes = recipes_df['id'].tolist()
all_ingr = recipes_df['ingredient_ids'].tolist()
print(len(all_recipes), len(all_ingr))

for i in range(len(all_recipes)):
    # {recipe: [[proteins], [produce], [dairy], [other]]}
    recipe = all_recipes[i]
    recipe_map[recipe] = [[],[],[], []]
    list_of_i = eval(all_ingr[i])
    for ingr in list_of_i:
        #split the ingredient and test each word in it (so chicken goes in no matter what?)
        if ingr in protein_ids:
            recipe_map[recipe][0].append(ingr)
        elif ingr in produce_ids:
            recipe_map[recipe][1].append(ingr)
        elif ingr in dairy_ids:
            recipe_map[recipe][2].append(ingr)
        else:
            recipe_map[recipe][3].append(ingr)
    
map = pd.DataFrame({'recipe_id': recipe_map.keys(), 'ingr_ids': recipe_map.values()})
map.to_csv("recipe_main_ingr_map.csv", index = True)
        

