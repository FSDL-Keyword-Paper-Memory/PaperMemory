import os
from argparse import Namespace

from flask import Flask, Response, jsonify, request

from predict import main

MODEL = os.getenv("MODEL")
THRESHOLD = float(os.getenv("THRESHOLD"))


app = Flask(__name__)


@app.route("/predict/", methods=["POST"])
def predict():
    content = request.get_json(force=True)
    abstract = content.get("abstract")

    if not abstract:
        return Response(status=400)

    namespace = Namespace(model=MODEL, threshold=THRESHOLD, abstract=abstract)
    keywords = main(namespace)

    return jsonify({"keywords": keywords})
