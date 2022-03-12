import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# load data
def load_data(folder):
    dir_list = os.listdir(folder)
    emotion = []
    file_path = []
    for file in dir_list:
        part = file.split('_')
        if part[2] == 'ANG':
            emotion.append(0)
        elif part[2] == 'DIS':
            emotion.append(1)
        elif part[2] == 'FEA':
            emotion.append(2)
        elif part[2] == 'HAP':
            emotion.append(3)
        elif part[2] == 'NEU':
            emotion.append(4)
        elif part[2] == 'SAD':
            emotion.append(5)
        else:
            emotion.append(6)
        file_path.append(folder + file)

    # create dataframe
    emotion_df = pd.DataFrame(emotion, columns=['label'])
    path_df = pd.DataFrame(file_path, columns=['path'])
    data_df = pd.concat([emotion_df, path_df], axis=1)
    data_df.to_csv('data_test.csv', index=False)
    print(data_df)
    print(emotion_df)

    return data_df, emotion_df


# Extract feature(MFCC,MEL,CONTRAST,CHROMA,RMS)
def extract_features(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - STFT (stft)
            - Spectral Contrast (contrast)
            - Chroma (chroma)
            - MFCC (mfcc)
            - Root Mean Square Value (rms)
            - MEL Spectrogram Frequency (mel)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    CONTRAST = kwargs.get("contrast")
    CHROMA = kwargs.get("chroma")
    MFCC = kwargs.get("mfcc")
    RMS = kwargs.get("rms")
    MEL = kwargs.get("mel")

    df = pd.DataFrame(columns=['feature'])
    counter = 0
    for label, path in enumerate(file_name.path):
        data, sample_rate = librosa.load(path, res_type='kaiser_fast', sr=None)
        result = np.array([])
        if CHROMA or CONTRAST:
            stft = np.abs(librosa.stft(data))
            if CONTRAST:
                contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, contrast))
            if CHROMA:
                chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma_stft))
        if MFCC:
            mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mfcc))
        if RMS:
            rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
            result = np.hstack((result, rms))
        if MEL:
            mel = np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        df.loc[counter] = [result]
        counter = counter + 1
    df = pd.DataFrame(df['feature'].values.tolist())
    df = df.fillna(0)
    return df


# train_test_split and then adjust dimensions
def tts_adjust(df, labels):
    x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, shuffle=True)
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    return x_train, x_test, y_train, y_test


# display gen graphs
def displayplot(cnn1, cnn2, label1, label2, title, x, y):
    plt.plot(cnn1, label=label1)
    plt.plot(cnn2, label=label2)
    plt.title(title)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.legend()
    plt.show()


# display accuracy
def displayplot_acc(cnn, title):
    plt.plot(cnn.history['accuracy'], label='Training Accuracy')
    plt.plot(cnn.history['val_accuracy'], label='Test Accuracy')
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


# data visualization
def displaydatapie():
    pic = pd.Series({'ANG':1271, 'DIS':1271, 'FEA':1271, 'HAP':1271, 'NEU':1087, 'SAD':1271})
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    lbs = pic.index
    explodes = [0.1 if i == '0' else 0 for i in lbs]
    plt.pie(pic, explode=explodes, labels=lbs, autopct="%1.1f%%", colors=sns.color_palette("muted"),
            startangle=90, pctdistance=0.6, textprops={'fontsize': 14, 'color': 'black'})
    plt.axis('equal')
    plt.show()
