import streamlit as st
import pandas as pd
import torch
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, TransformerDocumentEmbeddings, ELMoEmbeddings, DocumentPoolEmbeddings




bert_embedding = TransformerDocumentEmbeddings('bert-base-uncased')

elmo_embedding = DocumentPoolEmbeddings([ELMoEmbeddings('')])

sentence1 = Sentence('The grass is green.')
sentence2 = Sentence('a random sentence with two drops of sugar.')


sent1 = st.text_input(f"Write a sentence", "The grass is green.")

sent2 = st.text_input(f"Write another sentence", "The grass is not green.")



    
    
#assert sentence1.embedding.shape == sentence2.embedding.shape











def runComp(s1, s2):
    a = Sentence(s1)
    b = Sentence(s2)
    
    
    bert_embedding.embed(a)
    bert_embedding.embed(b)
    cos1 = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    prox1 = cos1(a.embedding, b.embedding)
    similarities1 = round(prox1.item(), 4)
    
    
    elmo_embedding.embed(a)
    elmo_embedding.embed(b)
    cos2 = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    prox2 = cos2(a.embedding, b.embedding)
    similarities2 = round(prox2.item(), 4)
    data = {'BERT':[similarities1], 'ELMo': [similarities2]}
    
    mydata = pd.DataFrame(data)
    
    st.text('Below you see the similarity between the two sentences you entered:')
    
    st.dataframe(mydata)
    
    return

#print('Similarity1: ')
#print(similarities1)

#print('Similarity2: ')
#print(similarities2)

if st.button('Run'):
    runComp(sent1, sent2)
    
#for token in sentence1:
#    print(token)
#    print(token.embedding)
    
#for token in sentence2:
#    print(token)
#    print(token.embedding)
