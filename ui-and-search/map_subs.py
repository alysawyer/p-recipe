import pandas as pd

from map_ingr import * 

ingr_map = pd.read_pickle('ingr_map.pkl')
recipes_df = pd.read_csv('PP_recipes.csv')


cooking_substitutions = {
    "all-purpose flour": {
        "substitute": [["whole wheat flour"]],
        "quantity": "1 cup",
        "notes": "For a healthier option, replace with whole wheat flour."
    },
    "baking powder": {
        "substitute": [["baking soda", "cream of tartar"]],
        "quantity": "1 tsp baking powder = 1/4 tsp baking soda + 1/2 tsp cream of tartar",
        "notes": "Mix the two ingredients together to replace baking powder."
    },
    "buttermilk": {
        "substitute": [["milk", "vinegar"], ["milk", "lemon juice"], ["yogurt"]],
        "quantity": "1 cup buttermilk = 1 cup milk + 1 tbsp vinegar or lemon juice = 1 cup yogurt",
        "notes": "Stir and let it sit for a few minutes before using."
    },
    "egg": {
        "substitute": [["applesauce"], ["banana"]],
        "quantity": "1 egg = 1/4 cup applesauce or mashed banana",
        "notes": "Use in recipes where egg is for moisture and binding."
    },
    "brown sugar": {
        "substitute": [["white sugar", "molasses"]],
        "quantity": "1 cup brown sugar = 1 cup white sugar + 1-2 tbsp molasses",
        "notes": "Adjust the amount of molasses based on desired sweetness and flavor."
    },
    "cornstarch": {
        "substitute": [["all-purpose flour"]],
        "quantity": "1 tbsp cornstarch = 2 tbsp all-purpose flour",
        "notes": "You can use flour in place of cornstarch for thickening."
    },
    "honey": {
        "substitute": [["maple syrup"]],
        "quantity": "1 cup honey = 1 cup maple syrup",
        "notes": "Maple syrup provides a similar level of sweetness."
    },
    "balsamic vinegar": {
        "substitute": [["red wine vinegar"]],
        "quantity": "1 tbsp balsamic vinegar = 1 tbsp red wine vinegar",
        "notes": "Red wine vinegar can be used as an alternative in recipes."
    },
    "cayenne pepper": {
        "substitute": [["paprika"]],
        "quantity": "1/4 tsp cayenne pepper = 1 tsp paprika",
        "notes": "Paprika is milder and can be used in place of cayenne."
    },
    "mayonnaise": {
        "substitute": [["greek yogurt"]],
        "quantity": "1 cup mayonnaise = 1 cup Greek yogurt",
        "notes": "Greek yogurt is a healthier alternative in many recipes."
    },
    "soy sauce": {
        "substitute": [["tamari sauce"]],
        "quantity": "Use the same quantity of tamari as soy sauce",
        "notes": "Tamari is a gluten-free alternative to soy sauce."
    },
    "oregano": {
        "substitute": [["marjoram"]],
        "quantity": "Use marjoram in the same quantity as oregano",
        "notes": "Marjoram has a similar flavor profile."
    },
    "parmesan cheese": {
        "substitute": [["pecorino romano cheese"]],
        "quantity": "Use Pecorino Romano in the same quantity as parmesan",
        "notes": "Pecorino Romano is a close alternative to parmesan."
    },
    "heavy cream": {
        "substitute": [["milk", "butter"]],
        "quantity": "1 cup heavy cream = 3/4 cup milk + 1/4 cup melted butter",
        "notes": "Mix the ingredients together for a reasonable alternative."
    }
    # Add more ingredients and their substitutions here
}


# substitute format {ingr: {'substitute': " (if + include both, if 'or' include one or other)", quantity: }}
substitution_map = {}
# we want a dictionary: {'ingr': {substitutes: [ands listed in here], (commas separate "ors")], instructions: ''}}
l_ingr_keys = list(cooking_substitutions.keys())
for ingr in l_ingr_keys:
    ingr_id = get_ingr_ids([ingr]) #this returns a list!!
    substitutions = cooking_substitutions[ingr]["substitute"]
    sub_options = []
    number_subs = len(substitutions)
    for i in range(number_subs): # for each set of substitutions (some have multiple)
        sub_ingr_ids = get_ingr_ids(substitutions[i]) #get ids for each ingr in a substitution
        if len(sub_ingr_ids) == len(substitutions[i]): #check that every ingr had a sub
            sub_options.append(sub_ingr_ids)
    if len(sub_options) > 0 and len(ingr_id) > 0:
        substitution_map[ingr_id[0]] = sub_options


map = pd.DataFrame({'ingr_id': substitution_map.keys(), 'substitutions': substitution_map.values()})
map.to_csv("ingr_subs.csv", index = True)

