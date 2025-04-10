import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
import numpy as np
from tensorflow.image import resize
import os

# do something about the file path of audio file

@st.cache_resource()
def load_model():
  model = tf.keras.models.load_model("Trained_model.h5")
  return model

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2 
                
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
        chunk = audio_data[start:end]
                    
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

if(app_mode=="Home"):
    st.markdown(
    """
    <style>
    h2, h3 {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown(''' ## Introducing the Music Genre Classification System! 🎶🎧''')
    st.markdown("""
**Our mission is to streamline the process of identifying music genres from audio tracks. Simply upload an audio file, and our advanced system will analyze it to determine its genre. Experience the innovative power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Visit the **Prediction** page and upload your audio file.    
2. **Analysis:** Our system uses cutting-edge algorithms to categorize the audio into one of the predefined genres.
3. **Results:** See the predicted genre along with additional relevant details.

### Why Choose Our Service?
- **High Accuracy:** We utilize the latest deep learning models to ensure precise genre predictions.
- **Easy to Use:** Our interface is designed to be simple and intuitive, providing a seamless user experience.
- **Quick Results:** Receive your results swiftly, allowing for faster music organization and discovery.
                
### Get Started Now
Navigate to the **Prediction** page in the sidebar, upload your audio file, and discover the capabilities of our Music Genre Classification System.

### About Us
Learn more about the project and our team on the **About Project** page.
""")

elif(app_mode=="About Project"):
    st.markdown("""
                ### About Project
                Music experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

                This data hopefully can give the opportunity to do just that.

                ### About Dataset
                #### Content
                1. **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
                2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
                3. **images original** - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
                4. **2 CSV files** - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.

                ### Approach
                1. **Audio Preprocessing:** Each audio file is divided into 4-second chunks with a 2-second overlap to capture diverse audio features.
                2. **Feature Extraction:** These chunks are converted into mel-spectrograms, which visually represent the audio's spectral characteristics.
                3. **Model Training:** The mel-spectrograms are fed into a Convolutional Neural Network (CNN) for training, leveraging the model's ability to recognize patterns in images.
                4. **Genre Classification:** The trained CNN model classifies the audio samples into their respective genres based on the learned patterns.
                
                This approach combines the strengths of audio signal processing and deep learning to achieve accurate music genre classification.

                ## Our Team
                1. **Himel Jana** (2205902)
                1. **Harsh Rastogi** (22051074)
                1. **Hrishant Atri** (22051077)
                1. **Himanshu Saini** (22051339)
                1. **Rahul Kumar Singh** (22051358)
                """)

elif(app_mode=="Prediction"):
    filepath=[] 
    st.header("Model Prediction")
    file_mode = st.selectbox("Select Type",["file", "folder"])

    if(file_mode=="file"):
        test_mp3_files = st.file_uploader("Upload an audio file", accept_multiple_files=True)
        if test_mp3_files is not None:
            for mp3_file in test_mp3_files: 
                temp_path = os.path.join("Test", mp3_file.name)
                os.makedirs("Test", exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(mp3_file.getbuffer())

                filepath.append(os.path.abspath(temp_path))

    elif(file_mode=="folder"):
        folder = st.text_input(label="Input folder path")
        if folder!="":
            for song_file in os.listdir(f'{folder}'):
                filepath.append(f'{folder}/{song_file}')
        
    if(st.button("Play Audio")):
        if len(filepath)!=0:
            for mp3_file in filepath:
                st.audio(mp3_file)
        else:
            st.error('Select an audio file', icon="🚨")
    
    if(st.button("Predict")):
      if len(filepath)!=0:
        with st.spinner("Please Wait.."):
            for path in filepath:       
                X_test = load_and_preprocess_data(path)
                result_index = model_prediction(X_test)
                st.balloons()
                label = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
                st.markdown("**:blue[Model Prediction:] {song_name} is a  :red[{result}] music**".format(song_name =path[5:], result=label[result_index]))
      else:
          st.error('Select an audio file or folder', icon="🚨")