from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from caption_generator import generate_caption

app = FastAPI()

class Item(BaseModel):
    name: str
    

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Route with a path parameter
@app.post("/get_caption")
def get_caption(item: Item):
    file_name = item.name
    print(file_name)
    cap, des = generate_caption(file_name)
    
    return {"cap" : cap , "des" : des}


if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=8000)
