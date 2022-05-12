import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import io

@st.cache(allow_output_mutation=True)
def load_model(modelname):
	return SentenceTransformer(modelname)

model = load_model('allenai-specter')

# workaround for when embeddings were done in GPU environment:
# https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
class CPU_Unpickler(pickle.Unpickler):
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
			return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
		else: return super().find_class(module, name)

@st.experimental_singleton
def load_data():
	with open('./data/hicss56_embeddings.pkl', "rb") as fIn:
		stored_data = CPU_Unpickler(fIn).load()
		minitracks = stored_data['minitracks']
		embeddings = stored_data['embeddings']
	return minitracks, embeddings

minitracks, minitrack_embeddings = load_data()

@st.experimental_singleton
def compute_minitrack_embeddings(minitracks):
	return model.encode(minitracks['description'], convert_to_tensor=True)

# model = SentenceTransformer('bert-base-nli-mean-tokens')
# model = SentenceTransformer('all-MiniLM-L6-v2')
# minitracks = fetch_and_clean_minitracks()
# minitrack_embeddings = compute_minitrack_embeddings(minitracks)

if 'abstract' not in st.session_state:
	st.session_state['abstract'] = ''

if 'results' not in st.session_state:
	st.session_state['results'] = None

#Compute cosine-similarities for each sentence with each other sentence
if len(st.session_state.abstract) > 0:
	abstract_embedding = model.encode([st.session_state.abstract], convert_to_tensor=True)
	cosine_scores = util.cos_sim(abstract_embedding, minitrack_embeddings)
	ordered = torch.argsort(cosine_scores[0], descending=True)
	results = []
	for idx, jTensor in enumerate(ordered):
		j = jTensor.item()
		if idx < 15:
			results.append({
				'rank': idx+1,
				'score': cosine_scores[0][j].item(),
				'track': f"{minitracks['trackname'][j]} ({minitracks['track'][j]})",
				'minitrack:': minitracks['name'][j],
			})
			#print(f"{minitracks['minitrackname'][j]} \t\t {cosine_scores[i][j]}")
	st.session_state['results'] = pd.DataFrame(results)

st.title("HICSS Minitrack Recommender")
st.header('Data')
st.write('Minitracks:')
# st.button('Update Minitracks', on_click=update_minitracks)
st.write(minitracks)

st.header('Input')
st.write("Input your abstract, title, or keywords and it the inputs similarity will be computed to all minitrack descriptions. ")
st.text_area("Your abstract", key="abstract", height=250)

st.header('Results')
st.write("The following ranked list shows the 15 minitracks with the highest cosine similarity score.")
# You can access the value at any point with:
if len(st.session_state.abstract) > 0:
	st.write(st.session_state['results'])

