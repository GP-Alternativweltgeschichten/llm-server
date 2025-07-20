from flask import Flask, jsonify, request
from model_registry import get_model

app = Flask(__name__)
model = get_model()

@app.route('/status')
def status():
    """Get server status."""
    return '{"status": "ok"}', 200, {'Content-Type': 'application/json'}


@app.route("/generate", methods=["POST"])
def generate():
    """Print available functions."""
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    response = model.generate(prompt)
    return jsonify({"response": response})


@app.route('/', methods=["GET"])
def help():
    """Print available functions."""
    func_list = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            func_list[rule.rule] = app.view_functions[rule.endpoint].__doc__
    return jsonify(func_list)


if __name__ == '__main__':
    app.run(threaded=True)
