import streamlit as st
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
	return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()

@st.experimental_memo
def fetch_and_clean_minitracks(_db_connection):
		# Fetch data from _db_connection here, and then clean it up.
		sql = """
			SELECT c.shortname as conference, t.code as track, t.name as trackname, m.id as id, m.name as name, m.description
			FROM minitracks m
			JOIN tracks t on t.id = m.track_id
			JOIN conferences c on c.id = t.conference_id
			where c.shortname = 'HICSS-56'
		"""
		minitracks = pd.read_sql_query(sql, _db_connection)
		return minitracks

@st.experimental_memo
def compute_minitrack_embeddings(_model, minitracks):
	return model.encode(minitracks['description'], convert_to_tensor=True)

@st.experimental_memo
def load_model(modelname):
	return SentenceTransformer(modelname)

# model = SentenceTransformer('bert-base-nli-mean-tokens')
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = load_model('allenai-specter')
minitracks = fetch_and_clean_minitracks(conn)
minitrack_embeddings = compute_minitrack_embeddings(model, minitracks)

if 'abstract' not in st.session_state:
	st.session_state['abstract'] = ''

if 'results' not in st.session_state:
	st.session_state['results'] = None

#Compute cosine-similarities for each sentence with each other sentence
if len(st.session_state.abstract) > 0:
	abstract_embedding = model.encode([st.session_state.abstract], convert_to_tensor=True)
	cosine_scores = util.cos_sim(abstract_embedding, minitrack_embeddings)
	cosine_scores.shape
	ordered = torch.argsort(cosine_scores[0], descending=True)
	results = []
	# ind = ordered[0:10]
	# results = minitracks[:][ind]
	print(minitracks.shape)
	print(ordered)
	for idx, jTensor in enumerate(ordered):
		j = jTensor.item()
		if idx < 10:
			results.append({'rank': idx+1, 'score': cosine_scores[0][j].item(), 'minitrack:': minitracks['name'][j]})
			#print(f"{minitracks['minitrackname'][j]} \t\t {cosine_scores[i][j]}")
	st.session_state['results'] = results

st.title("HICSS Minitrack Recommender")
st.header('Data')
st.write('Minitracks:')
st.write(minitracks)

st.header('Input')
st.write("Input your abstract: ")
st.text_area("Your abstract", key="abstract")

st.header('Results')
# You can access the value at any point with:
if len(st.session_state.abstract) > 0:
	st.write(st.session_state['results'])

