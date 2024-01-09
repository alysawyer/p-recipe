
# P-Recipe

Identifying viable recipes from a picture of ingredients. Full project specification available [here](https://docs.google.com/document/d/1Z-IRTpez5aXF5pA3BzhH3qOOYFAwqvfhkIYvM87OLm4/edit#heading=h.60k1wnj5cckc).

Project sponsored by the Claremont College's [P-ai club](https://www.p-ai.org/) during the 2023 fall semester. 

## Authors

- [@tyoo2025](https://www.github.com/tyoo2025)

- [@jacksusank](https://www.github.com/jacksusank)

- [@bengisublr](https://www.github.com/bengisublr)

- [@ayjj2023](https://www.github.com/ayjj2023)

- [@alysawyer](https://www.github.com/alysawyer)

## Demo

Insert gif or link to demo


## Features

- Fully functional UI that takes in an image and utilizes a computer vision model to identify the foods in the image
- Based on the foods in the image, recipes are suggested for the user
- Users can specify what foods to avoid, maximum time limit to cook when searching for recipes
- Recipe results can be filtered by amount of time to make it and sugars/fats 
- Additional ingredients can be inputted 

## Installation

Below is a sample process that can be used to get the program working: 

1. First, install the following files from [this](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) recipe dataset: ingr_map.pkl, PP_recipes.csv, RAW_recipes.csv, and RAW_interactions.csv. Installation of [Python](https://www.python.org/downloads/) and [Gradio](https://www.gradio.app/guides/installing-gradio-in-a-virtual-environment) is also necessary. 

2. Next, clone the repo: 
```bash
  git clone https://github.com/alysawyer/p-recipe.git
```

3. Navigate to the ui-and-search folder, and run the following files to set up the program as shown below:
```bash
  cd ui-and-search
  python map_recipe_ingr.py
  python map_subs.py
```

4. Finally, run this command to start the program:
```bash
  python ui_base.py
```
