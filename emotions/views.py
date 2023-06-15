from django.contrib.auth.decorators import login_required
from django.shortcuts import render

import tensorflow as tf
import tensorflow_io as tfio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

# Load the TensorFlow model
reloaded_model = tf.saved_model.load('model')


# my_classes = ['Discomfort', 'Burping', 'Tired', 'Belly_pain', 'Hungry']


@tf.function
def load_wav_16k_mono(file_contents):
    """ read in a waveform file and convert to 16 kHz mono """
    wav, sample_rate = tf.audio.decode_wav(
        file_contents.read(),
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


@login_required(login_url='user_creation:login')
def predict_emotion(request):
    # Handle audio input from the request
    if request.method == 'POST':
        audio_data = request.FILES.get('audio')

        # Preprocess and extract relevant features from the audio data
        if audio_data:
            reloaded_results = reloaded_model(load_wav_16k_mono(audio_data))

            my_classes = ['Discomfort', 'Burping', 'Tired', 'Belly_pain', 'Hungry']
            result = my_classes[tf.argmax(reloaded_results)]
            probs = tf.nn.softmax(reloaded_results) * 100
            probs = tf.nn.softmax(reloaded_results)
            plt.bar(my_classes, probs)
            plt.savefig('graph.png')
            plt.clf()

            # Encode the image data as base64
            with open('graph.png', 'rb') as img_file:
                image_data = img_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')

            context = {
                'predicted_emotion': result,
                'emotion_graph': image_base64
            }

            return render(request, 'index.html', context)
        else:
            # Audio file is not uploaded
            error_message = 'Please upload an audio sample.'
            context = {
                'error_message': error_message,
            }
            return render(request, 'index.html', context)

    # Return the predicted emotion as a response
    return render(request, 'index.html')


def home(request):
    return render(request, 'index.html', {})
