import pandas as pd

dataset_path = "C:/Users/Yuri/Documents/VSCode/dataset.csv"
music_data = pd.read_csv(dataset_path)

metal_music = music_data[music_data['track_genre'].str.contains("metal", case=False, na=False)]

print(metal_music.head())
