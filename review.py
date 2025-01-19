import torch
from transformers import pipeline

BASE_MODEL_NAME = "albert-base-v2"
FINETUNED_MODEL_PATH = "./training/steam_review_model"

class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

device = 0 if torch.cuda.is_available() else -1
torch_d_type = torch.float16 if torch.cuda.is_available() else torch.float32

classifier = pipeline(
    task="text-classification",
    model=FINETUNED_MODEL_PATH,
    tokenizer=BASE_MODEL_NAME,
    device=device,
    top_k=None,
    truncation=True,
    max_length=512,
    torch_dtype=torch_d_type)


reviews = [
    "Good game. Gunplay is enjoyable and both Warfare and Operations modes are fun to play. There are still some problems like running into cheaters more often, but other than that the game is really enjoyable. Plus it's free-to-play so give it a try!",
    "The game is absolute trash. Uninstalled.",
    "This game is incredible until Jeff eats your entire team.",
    "I will eat 1 tablespoon of Mayonnaise for every like this review gets.",
    "Nice strategy game with a cute and friendly community. It's good opportunity to learn a lot about your mother.",
    "Let me get this straight, you want me to tell what I think about this game? Well, you've come to the wrong place. However, if you have come for an outstanding breakfast pancake recipe, you've come to the right place: Ingredients (Serves 4): - 1 cup all-purpose flour - 2 tablespoons sugar - 1 teaspoon baking powder - 1/2 teaspoon baking soda - 1/4 teaspoon salt - 3/4 cup milk (or buttermilk) - 1 large egg - 2 tablespoons melted butter or vegetable oil - 1 teaspoon vanilla extract (optional) Instructions: - Mix Dry Ingredients: In a large bowl, whisk together the flour, sugar, baking powder, baking soda, and salt. - Mix Wet Ingredients: In another bowl, whisk the milk, egg, melted butter, and vanilla extract. - Combine: Gradually add the wet ingredients to the dry ingredients, stirring gently until just combined. (Some lumps are okay—don’t overmix!) - Heat the Pan: Heat a non-stick skillet or griddle over medium heat. Lightly grease with butter or oil. - Cook Pancakes: Pour about 1/4 cup of batter onto the pan for each pancake. Cook until bubbles form on the surface and the edges look set (about 2 minutes). Flip and cook the other side until golden (about 1-2 minutes). - Serve: Stack the pancakes and serve warm with your favorite toppings like syrup, fresh fruit, or whipped cream."
]

for review in reviews:
    result = classifier(review)[0]

    print("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║ Review: " + review.ljust(97)[:97] + "║")

    if result[0]['label'] == 'LABEL_1':
        print(f"║ Result: {bcolors.OKGREEN}Review is helpful.{bcolors.ENDC} Helpful Score: {round(result[0]['score'], 5)} | Unhelpful Score: {round(result[1]['score'], 5)}".ljust(115)[:115] + " ║")
    else:
        
        print(f"║ Result: {bcolors.WARNING}Review is NOT helpful.{bcolors.ENDC} Helpful Score: {round(result[1]['score'], 5)} | Unhelpful Score: {round(result[0]['score'], 5)}".ljust(115)[:115] + " ║")

    print("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
