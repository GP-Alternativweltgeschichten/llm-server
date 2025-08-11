import json
import shutil

from flask import Flask, jsonify, request, send_file
from model_registry import get_model, get_blender_model, get_3d_model
import subprocess
import uuid
import os

app = Flask(__name__)
model = get_model()
# blender_model = get_blender_model()
model_3d = get_3d_model()

@app.route('/status')
def status():
    """Get server status."""
    return '{"status": "ok"}', 200, {'Content-Type': 'application/json'}


@app.route("/generate", methods=["POST"])
def generate():
    """Print available functions."""
    data = request.get_json()
    print("Received data:", data)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    response = model.generate(prompt)
    return jsonify({"response": json.loads(response)})

@app.route("/generate_blender_code", methods=["POST"])
def generate_blender_code():
    """Generate Blender-compatible Python script and run it."""
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        result, output_model_path = generate_model_with_blender(prompt)

        if result.returncode != 0:
            return jsonify({
                "error": "Blender execution failed",
                "stderr": result.stderr,
                "stdout": result.stdout
            }), 500

        return send_file(
            output_model_path,
            as_attachment=True,
            download_name="generated_model.fbx",
            mimetype="application/octet-stream"
        )

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Blender process timed out"}), 500

@app.route("/generate_3d_model", methods=["POST"])
def generate_3d_model():
    """Generate 3D model using ShapE."""
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        output_model_path = model_3d.generate(prompt)
        return send_file(
            output_model_path,
            as_attachment=True,
            download_name="generated_model.obj",
            mimetype="application/octet-stream"
        )
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

def generate_model_with_blender(prompt):
    blender_script = blender_model.generate(prompt)
    unique_id = str(uuid.uuid4())
    os.makedirs("./output", exist_ok=True)

    script_path = f"./output/blender_script_{unique_id}.py"
    output_model_path = os.path.abspath(f"./output/generated_model_{unique_id}.fbx")
    with open(script_path, "w") as f:
        f.write(blender_script)

    result = subprocess.run(
        ["blender", "--background", "--python", script_path, "--", output_model_path],
        capture_output=True, text=True, timeout=30
    )
    print(result.stdout)
    print(result.stderr)
    return result, output_model_path

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
