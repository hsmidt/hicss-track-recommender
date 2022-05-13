import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import io

# @st.cache(allow_output_mutation=True)
@st.experimental_memo
def load_model(modelname):
	return SentenceTransformer(modelname)

# Need to keep it simpler due to limited resources with the Streamlit Cloud instance
# model_names = ['allenai-specter', 'all-mpnet-base-v2']
model_names = ['allenai-specter']
default_model = 'allenai-specter'
if 'model' not in st.session_state:
	st.session_state['model'] = load_model(default_model)
if 'modelname' not in st.session_state:
	st.session_state['modelname'] = default_model

def update_model(modelname):
	print(modelname)
	st.session_state['modelname'] = modelname
	print(st.session_state['modelname'])
	st.session_state['model'] = load_model(modelname)

# workaround for when embeddings were done in GPU environment:
# https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
class CPU_Unpickler(pickle.Unpickler):
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
			return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
		else: return super().find_class(module, name)

@st.experimental_singleton
def load_data():
	with open('./data/hicss56_embeddings_2.pkl', "rb") as fIn:
		stored_data = CPU_Unpickler(fIn).load()
		minitracks = stored_data['minitracks']
		embeddings = stored_data['embeddings']
	return minitracks, embeddings

minitracks, minitrack_embeddings = load_data()

if 'abstract' not in st.session_state:
	st.session_state['abstract'] = ''

if 'results' not in st.session_state:
	st.session_state['results'] = None

if 'top_k' not in st.session_state:
	st.session_state['top_k'] = 20

def update_topk(num):
	st.session_state['top_k'] = num

def get_result_df(search_results, results):
  return pd.DataFrame(
		[(score, mtrack, track) for score, mtrack, track in zip(
			[x['score'] for x in search_results[0]], results['name'].values, results['track']
		)], columns=['score', 'minitrack', 'track'])

#Compute cosine-similarities for each sentence with each other sentence
if len(st.session_state.abstract) > 0:
	abstract_embedding = st.session_state['model'].encode([st.session_state.abstract], convert_to_tensor=True)
	# cosine_scores = util.cos_sim(abstract_embedding, minitrack_embeddings)
	search_results = util.semantic_search(abstract_embedding, minitrack_embeddings[st.session_state['modelname']], top_k=st.session_state['top_k'])
	search_ind = [x['corpus_id'] for x in search_results[0]]
	mtracks = minitracks.iloc[search_ind]
	st.session_state['results'] = get_result_df(search_results, mtracks)

st.title("HICSS Minitrack Recommender")
st.header('Data')
st.write('Minitracks:')
# st.button('Update Minitracks', on_click=update_minitracks)
st.dataframe(minitracks)

st.header('Minitrack Recommender')
st.subheader('Model Selection')
st.write('Current model:', st.session_state['modelname'])

model_selection = st.radio(
	'Select the pre-trained model that you want to use.',
	model_names, 
	index=model_names.index(st.session_state['modelname']))
st.button('Update Model', on_click=update_model, args=(model_selection,))

st.subheader('Input')
st.write("Input your abstract, title, or keywords and it the inputs similarity will be computed to all minitrack descriptions. ")
st.text_area("Your abstract", key="abstract", height=250)

st.header('Results')
st.write(f"The following ranked list shows the {st.session_state['top_k']} minitracks with the highest cosine similarity score.")
slider_topk = st.slider(
     'Select number of desired results',
     5, 30, st.session_state['top_k'])
st.button('Update Result Count', on_click=update_topk, args=(slider_topk,))

# You can access the value at any point with:
if len(st.session_state.abstract) > 0:
	st.table(st.session_state['results'])

