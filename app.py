from algorithms.recommender import Recommender
from flask import Flask, jsonify
from flask_restful import Api
from flask_cors import CORS

app = Flask(__name__)
app.config['DEBUG'] = True
api = Api(app)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/recommendations/user/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    res = Recommender.get_top_n_for_user(uid = user_id)
    print(res)
    return jsonify(res)

if __name__ == '__main__':
    app.run(port = 5000)  # important to mention debug=True
