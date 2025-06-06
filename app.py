import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix
import spotipy
from spotipy.oauth2 import SpotifyOAuth

st.set_page_config(page_title="Music Recommender", layout="wide")

CLIENT_ID = st.secrets["spotify"]["client_id"]
CLIENT_SECRET = st.secrets["spotify"]["client_secret"]
REDIRECT_URI = st.secrets["spotify"]["redirect_uri"]
SCOPE = "playlist-modify-public playlist-modify-private"

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
)

query_params = st.query_params
code = query_params.get("code", [None])[0]

if 'token_info' not in st.session_state:
    if code:
        token_info = sp_oauth.get_access_token(code=code, as_dict=True)
        st.session_state.token_info = token_info
        st.query_params.clear() 
        st.success("✅ Logged in successfully with Spotify!")
    else:
        auth_url = sp_oauth.get_authorize_url()
        st.markdown(f"[🔑 Login with Spotify]({auth_url})")
        st.stop()

token_info = st.session_state.get("token_info")
sp = spotipy.Spotify(auth=token_info['access_token'])

df = pd.read_csv('preprocessed_dataset.csv')
df['artists'] = df['artists'].astype(str)
df['track_genre'] = df['track_genre'].astype(str)
df['track_name'] = df['track_name'].astype(str)

numerical_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'liveness', 'speechiness', 'instrumentalness']
scaler = MinMaxScaler()
normalized_numerical = scaler.fit_transform(df[numerical_features])
numerical_sparse = csr_matrix(normalized_numerical)

genre_vectorizer = TfidfVectorizer()
genre_features = genre_vectorizer.fit_transform(df['track_genre'])

artist_vectorizer = TfidfVectorizer()
artist_features = artist_vectorizer.fit_transform(df['artists'])

num_w, gen_w, art_w = 1.0, 1.5, 0.5
combined_features = hstack([
    numerical_sparse * num_w,
    genre_features * gen_w,
    artist_features * art_w
])

knn_model = NearestNeighbors(n_neighbors=51, metric='cosine')
knn_model.fit(combined_features)

def content_based_recommend(song_index, top_n=50):
    distances, indices = knn_model.kneighbors(combined_features[song_index], n_neighbors=top_n + 1)
    recommended_indices = indices[0][1:]
    return df.iloc[recommended_indices][['track_name', 'artists', 'track_genre', 'popularity']]

def recommend_by_song_name(song_name, top_n=50):
    matches = df[df['track_name'].str.contains(song_name, case=False, na=False)]
    if matches.empty:
        return f"No song found with name similar to '{song_name}'"
    song_index = matches.index[0]
    return content_based_recommend(song_index, top_n)

def get_spotify_track_id(sp, track_name, artist_name):
    main_artist = artist_name.split(";")[0].strip()
    query = f"track:{track_name} artist:{main_artist}"
    try:
        results = sp.search(q=query, type='track', limit=1)
        tracks = results.get('tracks', {}).get('items', [])
        if tracks:
            return tracks[0]['id']
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify search failed for '{track_name}' by '{main_artist}': {e}")
    return None

def recommend_and_create_playlist(song_name, top_n=50):
    matches = df[df['track_name'].str.contains(song_name, case=False, na=False)]
    if matches.empty:
        return f"No song found with name similar to '{song_name}'"
    song_index = matches.index[0]
    recommendations = content_based_recommend(song_index, top_n)

    user_id = sp.current_user()['id']
    playlist_name = f"Recommendations Based on {df.loc[song_index, 'track_name']}"
    playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=False)
    playlist_id = playlist['id']

    track_uris = []
    for _, row in recommendations.iterrows():
        track_id = get_spotify_track_id(sp, row['track_name'], row['artists'])
        if track_id:
            track_uris.append(f"spotify:track:{track_id}")

    if not track_uris:
        return "No valid Spotify tracks found."

    for i in range(0, len(track_uris), 100):
        sp.playlist_add_items(playlist_id=playlist_id, items=track_uris[i:i+100])

    return f"Playlist '{playlist_name}' created with {len(track_uris)} tracks."


st.title("🎧 Music Recommendation System")

df['dropdown_label'] = df['track_name'] + " - " + df['artists']
selected_song = st.selectbox("🎵 Choose a song to get recommendations:", df['dropdown_label'].unique())

top_n = st.slider("📊 Number of recommendations", min_value=5, max_value=50, value=10, step=5)

track_name, artist_name = selected_song.split(" - ", 1)
matches = df[(df['track_name'] == track_name) & (df['artists'] == artist_name)]

if matches.empty:
    st.error("No matching song found in the dataset.")
    st.stop()

song_index = matches.index[0]

if st.button("Get Recommendations"):
    result = recommend_by_song_name(track_name, top_n)
    if isinstance(result, str):
        st.warning(result)
    else:
        st.dataframe(result)

if st.button("Create Playlist on Spotify"):
    with st.spinner("Creating playlist..."):
        result = recommend_and_create_playlist(track_name, top_n)
    if result:
        st.success(result)
