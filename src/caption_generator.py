from art_emotion_description import describe_image
from openai import OpenAI

import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

client = OpenAI( api_key=config["openai_key"])

def generate_caption(file_name):
    
    image_path = f"sample_img/{file_name}"
    art, emotion = describe_image(image_path)
    
    if len(art) == 0 or len(emotion) == 0:
        return "error in image use some other image" , "error"
    
    art_str = ", ".join(art)
    emotion_str = ", ".join(emotion)
    
    print(art_str)
    print(emotion_str)
    
    messages=[
        {
            "role": "system",
            "content": "You will be provided with some keywords, and your task is to create a generate a artistic caption and description of art in easy words. Given the keywords extracted from the image in 3 categories: Art Type, emotion of image and objects in the picture. Give me caption and description seperated by '\n\n' Do not give title."
        },
        {
        "role": "user",
        "content": "Art Type : {} \n Emotions : {}".format(art_str , emotion_str)
        }
    ]
    
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.8,
        max_tokens=100,
        #top_p=1
    )

    ans = result.choices[0].message.content.split("\n\n")
    print(ans)
    if len(ans) < 2:
        return "error in image use some other image" , "error"
    cap , des = ans
    cap = cap.split(":")[-1].strip()
    des = des.split(":")[-1].strip()

    return cap, des