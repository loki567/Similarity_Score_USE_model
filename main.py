from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

model = tf.saved_model.load("USE_model")

@app.route('/', methods=["POST"])
def calculate_similarity():
    # Get the request JSON data
    request_data = request.get_json()
    
    # Extract text1 and text2 from the request data
    text1 = request_data.get("text1")
    text2 = request_data.get("text2")
    
    # Encode the sentences using the loaded model
    embeddings = model([text1, text2])
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])
    
    # Normalize the similarity score to the range [0, 1]
    score = (similarity_score[0][0] + 1) / 2
    
    # Create the response JSON
    response_data = {"similarity_score": score}
    
    # Return the response as JSON
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)
