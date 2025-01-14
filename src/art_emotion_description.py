import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Vastly expanded seed words for arts and emotions
art_words = [
    "painting", "sculpture", "dance", "poetry", "theater", "music", "film",
    "photography", "literature", "drama", "opera", "design", "architecture",
    "drawing", "sketching", "ceramics", "fashion", "performance", "collage",
    "installation", "media", "animation", "printmaking", "exhibition", "gallery",
    "craft", "folk art", "modernism", "impressionism", "surrealism", "portrait",
    "landscape", "abstract", "graphic design", "illustration", "calligraphy",
    "mosaic", "street art", "graffiti", "carving", "digital art", "visual arts",
    "cinematography", "scriptwriting", "choreography", "composing", "orchestra",
    "pottery", "musical", "prose", "fine arts", "avant-garde", "expressionism",
    "realism", "cubism", "conceptual art", "baroque", "dadaism",
    "pop art", "renaissance", "romanticism", "neoclassicism", "futurism", "symbolism",
    "installation", "print", "still life", "sculpting", "impression", "etching",
    "mixed media", "urban art", "live performance", "gallery installation",
    # Adding new words related to other art fields
    "ballet", "jazz", "acrylic", "watercolor", "pottery", "folk", "animation",
    "improvisation", "ink", "scenography", "symphony", "scoring", "arrangement",
    "literary", "sonnet", "metaphor", "narrative", "allegory", "pastel",
    "fresco", "stanza", "chorus", "conductor", "orchestration", "scene"
]

emotion_words = [
    "joy", "anger", "fear", "love", "sadness", "hope", "happiness", "grief",
    "surprise", "disgust", "envy", "pride", "contentment", "anxiety",
    "compassion", "desire", "nostalgia", "frustration", "excitement",
    "affection", "tension", "tranquility", "confidence", "inspiration",
    "loneliness", "anticipation", "sorrow", "guilt", "shame", "melancholy",
    "boredom", "indifference", "euphoria", "ecstasy", "rage", "sympathy",
    "irritation", "resentment", "admiration", "awe", "gratitude", "relief",
    "tenderness", "self-esteem", "self-doubt", "motivation", "calmness",
    "apprehension", "satisfaction", "disappointment", "humiliation", "forgiveness",
    "hatred", "desperation", "insecurity", "empathy", "bitterness",
    "contentment", "vulnerability", "serenity", "compassion", "ecstasy", "remorse",
    "mourning", "relaxation", "sensitivity", "romanticism", "friendship",
    "jealousy", "arousal", "optimism", "faith", "regret", "obsession", "desperation",
    "resentment", "irritation", "joyfulness", "melancholia", "abandonment",
    "elation", "pity", "enlightenment", "foreboding", "calm", "reluctance",
    "jealousy", "appreciation", "patience", "distrust", "tolerance", "apathy",
    "compassion", "cynicism", "endearment", "repentance", "optimism"
]

objects = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", "Traffic light",
    "Fire hydrant", "Stop sign", "Parking meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep",
    "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase",
    "Frisbee", "Skis", "Snowboard", "Sports ball", "Kite", "Baseball bat", "Baseball glove", "Skateboard",
    "Surfboard", "Tennis racket", "Bottle", "Wine glass", "Cup", "Fork", "Knife", "Spoon", "Bowl",
    "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", "Hot dog", "Pizza", "Donut", "Cake",
    "Chair", "Couch", "Potted plant", "Bed", "Dining table", "Toilet", "TV", "Laptop", "Mouse", "Remote",
    "Keyboard", "Cell phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock",
    "Vase", "Scissors", "Teddy bear", "Hair dryer", "Toothbrush", "Drone", "Scooter", "Stroller", "Tablet",
    "Keyboard", "Desk", "Trash can", "Headphones", "Pencil", "Marker", "Speaker", "Calculator", "Magazine",
    "Newspaper", "Laptop charger", "Gloves", "Helmet", "Shovel", "Flashlight", "Fan", "Toolbox", "Chainsaw",
    "Backhoe", "Excavator", "Bulldozer", "Crane", "Wheelbarrow", "Lawn mower", "Rake", "Broom", "Mop",
    "Fire extinguisher", "Water bottle", "Lunchbox", "Wrench", "Drill", "Screwdriver", "Paintbrush",
    "Hammer", "Battery", "Bucket", "Soap", "Towel", "Watch", "Bracelet", "Earrings", "Rug", "Candle",
    "Curtain", "Mirror", "Shoe", "Slippers", "Socks", "Sandals", "Boots", "Cap", "Hat", "Sunglasses",
    "Watering can", "Laundry basket", "Pillow", "Blanket", "Plush toy", "Action figure", "Golf club",
    "Fishing rod", "Compass", "Binoculars", "Luggage", "Road sign", "Mailbox", "Traffic cone", "Tool belt",
    "Measuring tape", "Garden hose", "Mailbox", "Playground slide", "Swing set"
]


def describe_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Tokenize text lists and prepare model input
    art_text_inputs = processor(text=art_words, return_tensors="pt", padding=True)
    emotion_text_inputs = processor(text=emotion_words, return_tensors="pt", padding=True)
    # object_text_inputs = processor(text=objects, return_tensors="pt", padding=True)


    # Get image and text embeddings
    image_features = model.get_image_features(**inputs)
    art_text_features = model.get_text_features(**art_text_inputs)
    emotion_text_features = model.get_text_features(**emotion_text_inputs)
    # object_text_features = model.get_text_features(**object_text_inputs)

    # Normalize features and calculate cosine similarity for each list
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    art_text_features = art_text_features / art_text_features.norm(dim=-1, keepdim=True)
    emotion_text_features = emotion_text_features / emotion_text_features.norm(dim=-1, keepdim=True)
    # object_text_features = object_text_features / object_text_features.norm(dim=-1, keepdim=True)

    art_similarity = (image_features @ art_text_features.T).squeeze(0)
    emotion_similarity = (image_features @ emotion_text_features.T).squeeze(0)
    # object_similarity = (image_features @ object_text_features.T).squeeze(0)

    # Get top 3 matches for both lists
    top_k = 3
    top_art = 2
    top_emotion = 5
    top_art_matches = torch.topk(art_similarity, top_art)
    top_emotion_matches = torch.topk(emotion_similarity, top_emotion)
    # top_object_matches = torch.topk(object_similarity, top_k)

    # Retrieve the top 3 art and emotion words
    top_art_descriptions = [art_words[idx] for idx in top_art_matches.indices]
    top_art_scores = [score.item() for score in top_art_matches.values]

    top_emotion_descriptions = [emotion_words[idx] for idx in top_emotion_matches.indices]
    top_emotion_scores = [score.item() for score in top_emotion_matches.values]

    # top_object_descriptions = [objects[idx] for idx in top_object_matches.indices]
    # top_object_scores = [score.item() for score in top_object_matches.values]

    # Display results
    # art_result = "\n".join([f"{i + 1}. {desc} " for i, (desc, score) in
    #                         enumerate(zip(top_art_descriptions, top_art_scores))])
    # emotion_result = "\n".join([f"{i + 1}. {desc} " for i, (desc, score) in
    #                             enumerate(zip(top_emotion_descriptions, top_emotion_scores))])
    # object_result = "\n".join([f"{i + 1}. {desc}" for i, (desc, score) in
    #                         enumerate(zip(top_object_descriptions, top_object_scores))])

    # art_result = "".join([f"{desc}, " for i, (desc, score) in
    #                         enumerate(zip(top_art_descriptions, top_art_scores))])
    art_result = [desc for desc, score in zip(top_art_descriptions, top_art_scores)]
    emotion_result = [desc for desc, score in zip(top_emotion_descriptions, top_emotion_scores)]
    # emotion_result = "".join([f"{desc}, " for i, (desc, score) in
    #                             enumerate(zip(top_emotion_descriptions, top_emotion_scores))])

    #return f"Top {top_art} Art Descriptions:\n{art_result}\n\nTop {top_emotion} Emotion Descriptions:\n{emotion_result}"
    return art_result, emotion_result


#image_path = "i10.jpg"
#art, emotion = describe_image(image_path)
#print("Art : ", art)
#print("Emotion : ", emotion)
