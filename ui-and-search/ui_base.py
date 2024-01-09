from recipe_pool import *

import gradio as gr
import numpy as np

from PIL import Image
from model import *


def print_recipe(recipe_pool,subs_needed):
    """Given recipe_pool (a dictionary of recipe IDs:score in our final pool)
    Display/Print 5 recipes for the user"""
    all_recipe_ids = raw_recipes_df['id'].tolist()
    recipe_ids_pool = list(recipe_pool.keys())
    recipe_counter = 1

    recipes_to_display = []
    
    #loop through the first five recipes (max)
    while recipe_counter <= 5 and recipe_counter<= len(recipe_ids_pool):
        recipe_report = "" #a string to collect results
        recipe_id = recipe_ids_pool[recipe_counter-1]
        num_rating, avg_rating = get_avg_rating(recipe_id)
        recipe_ind = all_recipe_ids.index(recipe_id)
        recipe_row = raw_recipes_df.iloc[recipe_ind]

        #Add recipe title
        recipe_report += map_recipe_id_name(recipe_id)
        recipe_report += f"\nMISSING {recipe_pool[recipe_id]} INGREDIENTS"
        using_subs = True
        subs_used = subs_needed[recipe_id]
        if subs_needed[recipe_id] == {}:
            using_subs = False
        recipe_report+=f"\nUsing SUBSTITUES: {using_subs}"
        for key, value in subs_used.items():
            recipe_report += f"\n  - Use {value} instead of {key}"
        #print ratings, number of ratings
        recipe_report += f'\n \n Rating: {avg_rating:2f} among {num_rating} users \n'
        # print time estimate
        recipe_report += f'{recipe_row["minutes"]} minutes'
        #print ingredients
        recipe_report += f'\nIngredients: {recipe_row["ingredients"]}\n'
        #print instructions
        recipe_report += "\n \nSteps: \n"
        instruction = eval(recipe_row['steps'])
        for i in range(len(instruction)):
            recipe_report += f"{i+1}. {instruction[i]}\n"
        recipes_to_display.append(recipe_report)
        recipe_counter += 1
    return recipes_to_display

def main_no_input(ingredients, allergens, max_time, sort_criteria):
    ingr_ids = get_ingr_ids(ingredients)
    if allergens == "NA":
        recipe_pool, subs_needed = get_recipe_pool(ingr_ids)
    else:
        allergy_list = allergens.split(', ')
        recipe_pool, subs_needed = get_recipe_pool(ingr_ids, allergy_list)
    filtered_pool = filter_pool(recipe_pool)
    ordered_pool = order_ratings(filtered_pool)

    if max_time != "0":
        ordered_pool = limit_time(ordered_pool, int(max_time))
    
    #Sort by additional criteria? 'Time' for results in increasing order of time. 'Sugars' for results in decreasing order of sugars. 'Fats' for results in decreasing order of fats. 'NA' for none.")
    if sort_criteria == "Time":
        ordered_pool = order_times(ordered_pool)
    elif sort_criteria == "Sugars":
        ordered_pool = order_sugars(ordered_pool)
    elif sort_criteria == "Fats":
        ordered_pool = order_fats(ordered_pool)



    return print_recipe(ordered_pool, subs_needed)

def process_image(image):
    #image = np.ndarray(image)
    return predict3(image)
    

with gr.Blocks() as demo:
    error_box = gr.Textbox(label="Error", visible=False)

    input_im = gr.Image(label="Upload Images of Items HERE!", type="numpy")
    ingr = gr.Textbox(label="Ingredients", info = "ex. apples, bananas, oranges")
    allergens = gr.Textbox(label="Foods to Avoid", info = "ex. almond, peanut")
    max_time = gr.Textbox(label="Max Time", info="If no limit, select 0!")
    sort_criteria = gr.Radio(["Time", "Sugars", "Fats", "NA"], label="Sort Criteria", info="How should we sort the recipes?")
    
    go_btn = gr.Button("Get recipes")

    with gr.Accordion("Results Loading...") as outputLong:
        results1 = gr.Textbox("Loading...", label="Recipe 1")
        results2 = gr.Textbox("", label="Recipe 2")
        results3 = gr.Textbox("", label="Recipe 3")
        results4 = gr.Textbox("", label= "Recipe 4")
        results5 = gr.Textbox("", label="Recipe 5")

    def submit(ingr, allergens, max_time, sort_criteria, pic):
        if len(ingr) == 0:
            return {error_box: gr.Textbox(value="Enter ingredients", visible=True)}
        elif len(allergens) == 0:
            return {error_box: gr.Textbox(value="Enter allergen. If none, enter NA", visible=True)}
        elif len(max_time) == 0:
            return {error_box: gr.Textbox(value="Enter max time. If none, enter 0", visible=True)}
        image_results = process_image(pic)
        ingr_list = ingr.split(', ')
        full_ingr = image_results + ingr_list
        result_recipes = main_no_input(full_ingr, allergens, max_time, sort_criteria)
        #result_recipes = []
        for i in range(5- len(result_recipes)):
            result_recipes.append("No additional Results")
        return {
            outputLong: gr.Column("Open for Recipe Results!"),
            results1: result_recipes[0],
            results2: result_recipes[1],
            results3: result_recipes[2],
            results4: result_recipes[3],
            results5: result_recipes[4]
        }

    go_btn.click(fn=submit, inputs=[ingr, allergens, max_time, sort_criteria, input_im], outputs=[outputLong, results1, results2, results3, results4, results5], api_name="recipe")

demo.launch()
