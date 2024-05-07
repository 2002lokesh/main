from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load JSON data and preprocess patterns
with open('intents.json', 'r') as file:
    intents = json.load(file)

labels = []
patterns = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        labels.append(intent['tag'])
        patterns.append(pattern.lower())  # Convert patterns to lowercase for case insensitivity

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

def get_most_similar_intent(user_input, vectorizer, X, labels):
    user_vector = vectorizer.transform([user_input.lower()])  # Convert user input to lowercase
    similarity_scores = cosine_similarity(user_vector, X)
    most_similar_index = np.argmax(similarity_scores)
    return labels[most_similar_index]

# Define request model
class Item(BaseModel):
    message: str

# Define FastAPI endpoint for POST method
@app.post("/chatbot/")
async def chatbot(item: Item):
    user_input = item.message
    intent = get_most_similar_intent(user_input, vectorizer, X, labels)
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            responses = intent_data['responses']
            return {"Chatbot": np.random.choice(responses)}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Change host to "0.0.0.0" for Azure deployment
