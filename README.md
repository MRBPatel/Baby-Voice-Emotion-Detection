# Baby Emotion Detection using audio

The Baby Emotion Detection from Audio project is a machine learning-based system designed to analyze and classify the emotions expressed in audio recordings of babies. The system can classify emotions into categories such as discomfort, hunger, burping, tiredness, and belly pain. This project leverages various technologies including Django, TensorFlow, NumPy, and Matplotlib to create a user-friendly web application for efficient emotion detection.

## Features

- **Audio Emotion Classification:** The system can analyze audio recordings of babies and classify the emotions expressed within them into categories like discomfort, hunger, burping, tiredness, and belly pain.

- **Web Application:** The project includes a web application built using Django, providing users with an accessible and intuitive interface to upload and analyze audio recordings.

- **Machine Learning:** TensorFlow is used for implementing machine learning models to perform audio emotion classification, ensuring accurate and reliable results.

- **Version Control:** Check the `requirements.txt` file for version control of the project's dependencies, ensuring compatibility and reproducibility.

## Installation

To set up and run the Baby Emotion Detection from Audio project, follow these steps:

1. Clone the project repository:

   ```
   git clone https://github.com/MRBPatel/baby_emotion.git
   ```

2. Navigate to the project directory:

   ```
   cd baby-emotion
   ```

3. Install the required dependencies using `pip` and the `requirements.txt` file:

   ```
   pip install -r requirements.txt
   ```

4. Run the Django development server:

   ```
   python manage.py runserver
   ```

5. Access the web application in your browser by visiting `http://localhost:8000`.

## Usage

1. Open the web application in your browser.

2. Upload an audio recording of a baby expressing emotion.

3. Submit the recording for analysis.

4. The system will process the audio and classify the baby's emotion into one of the predefined categories.

5. View the results on the web interface, which may include visualizations generated using Matplotlib for a better understanding of the analysis.

## Contributing

Contributions to this project are welcome! If you have ideas for improvements, additional features, or bug fixes, please feel free to open an issue or submit a pull request.
