from flask import Flask, request, render_template
import logging
import sys
import traceback

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

# Configure logging
logging.basicConfig(level=logging.DEBUG)

application = Flask(__name__)
app = application
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Ensure templates reload without restart

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # No results on first visit
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            pred_df = data.get_data_as_data_frame()
            app.logger.debug(f"Input DataFrame:\n{pred_df}")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            app.logger.debug(f"Prediction Result: {results}")

            return render_template('home.html', results=round(results[0], 2))

        except Exception as e:
            error_trace = traceback.format_exc()
            app.logger.error(f"Exception during prediction:\n{error_trace}")
            return f"""
            <h1>Internal Server Error</h1>
            <h3>The server encountered an error while processing your request.</h3>
            <pre>{error_trace}</pre>
            """

if __name__ == "__main__":
    app.run(host="0.0.0.0")
