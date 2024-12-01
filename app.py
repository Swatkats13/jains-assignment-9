from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from neural_networks import visualize
import logging

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        # Parse the input parameters from the POST request
        activation = request.json['activation']
        lr = float(request.json['lr'])
        step_num = int(request.json['step_num'])

        # Log the parameters
        logging.debug(f"Received parameters: activation={activation}, lr={lr}, step_num={step_num}")

        # Run the visualization function
        visualize(activation, lr, step_num)

        # Verify that the result GIF is generated
        result_gif = os.path.join("results", "visualize.gif")
        if os.path.exists(result_gif):
            logging.debug("GIF generated successfully.")
            return jsonify({"result_gif": result_gif})
        else:
            logging.error("GIF file not found after running the experiment.")
            return jsonify({"error": "GIF file not generated"}), 500

    except Exception as e:
        # Log the exception
        logging.exception("Error occurred while running the experiment:")
        return jsonify({"error": str(e)}), 500

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    try:
        return send_from_directory('results', filename)
    except Exception as e:
        logging.exception("Error serving the file:")
        return jsonify({"error": str(e)}), 500

# Main entry point of the app
if __name__ == '__main__':
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Run the app in debug mode
    app.run(debug=True)
