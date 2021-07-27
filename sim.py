import streamlit as st
import torch
import plotly.graph_objs as go
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, TransformerDocumentEmbeddings, ELMoEmbeddings, DocumentPoolEmbeddings, WordEmbeddings

from stability import similarWords, stability
from nltk.corpus import brown
from gensim.models import word2vec
#### Static Word Embeddings Code below ####
def calcSim(exampleWord):
    sentences = brown.sents()
    sentences = [[word.lower() for word in sentence] for sentence in sentences]
    
    model1 = word2vec.Word2Vec(sentences, size=100,window=5,min_count=1, seed=42)
    model2 = word2vec.Word2Vec(sentences, size=100,window=5,min_count=1, seed=102)
    
    all_words = set([word for sentence in sentences for word in sentence])
    
    model1_dict = {}
    model2_dict = {}
    
    for word in all_words:
        model1_dict[word] = model1.wv[word]
        model2_dict[word] = model2.wv[word]
    
    mostSimilar1 = similarWords(model1_dict,exampleWord)
    mostSimilar2 = similarWords(model2_dict,exampleWord)
    
    stab = stability(exampleWord,[mostSimilar1,mostSimilar2],[mostSimilar1,mostSimilar2],True)
    st.text('"' + exampleWord + '" has a sability of ' + str(stab*10) + '%')
    return stab
    
main_word = st.text_input(f"Write a word","president")
if st.button('Show Stability'):
    calcSim(main_word)

#### Contextual Word Embeddings Code below ####
embedding_list = {'word':'glove', 'elmo':'','flair':'mix', 'bert':'bert-base-uncased','gpt':'openai-gpt','gpt2':'gpt2','roberta':'distilroberta-base'}

example_sent = ["the doctor invited the patient for lunch",
                "the doctor didn't invite the patient for lunch",
                "the patient invited the doctor for lunch",
                "the surgeon invited the patient for lunch",
                "the doctor invited the patient for a meal",
                "the doctor and the patient went our for tea",
                "for patient the invited doctor lunch the",
                "a random sentence with two drops of sugar",
                "esta frase está en otro idioma"]

def load_embeddings(etype):
    ename = embedding_list[etype]
    if etype == 'word':
        return DocumentPoolEmbeddings([WordEmbeddings(ename)])
    if etype == 'flair':
        return DocumentPoolEmbeddings([FlairEmbeddings(f'{ename}-forward'), FlairEmbeddings(f'{ename}-backward')])
    elif etype == 'elmo':
        return DocumentPoolEmbeddings([ELMoEmbeddings(ename)])
    elif etype == 'bert':
        return TransformerDocumentEmbeddings(ename, layers='-1,-2,-3,-4')
    elif etype == 'gpt':
        return TransformerDocumentEmbeddings(ename, layers='-1,-2,-3,-4')
    elif etype == 'gpt2':
        return TransformerDocumentEmbeddings(ename, layers='-1,-2,-3,-4')
    elif etype == 'roberta':
        return TransformerDocumentEmbeddings(ename, layers='-1,-2,-3,-4')
    else: 
        st.write('Error')
        
cols = st.beta_columns(len(embedding_list))
check_boxes = [cols[i].checkbox(name) for i, name in enumerate(embedding_list)]


main_sentence = st.text_input(f"Write a sentence", "The grass is green.")
similar_sentence = []
for i in range(1, len(example_sent[1:]), 2):
        # display sample sentences in double column format for better readability
        cols = st.beta_columns(2)
        similar_sentence.append(cols[0].text_input(f"Index {i-1}", example_sent[i], key=i))
        similar_sentence.append(cols[1].text_input(f"Index {i}", example_sent[i+1], key=i+1))


def plot(similarities, checked_names):
    layout = go.Layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='Similar sentence indexes: [0-7]')),
                       yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='Embedding type')))
    st.plotly_chart(go.Figure(data=go.Heatmap(z=similarities, y=checked_names, colorscale='Blues'), layout=layout))
    return

def runComp(main, similar, embeddings):
    similarities = []
    a = Sentence(main)
    
    embeddings.embed(a)
    
    for sent in similar:
        if not sent:
            continue
        b = Sentence(sent)
        embeddings.embed(b)
        
        cos1 = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        prox1 = cos1(a.embedding, b.embedding)
        similarities.append(round(prox1.item(), 4))
        
        
    return similarities
    



if st.button('Run'):
    checked_names = []
    similarities = []
    for name, box in zip(embedding_list, check_boxes):
        if box:
            checked_names.append(name)
            similarities.append(runComp(main_sentence, similar_sentence, load_embeddings(name)))
        
    if similarities:
        plot(similarities, checked_names)
    
        
            
    
    
