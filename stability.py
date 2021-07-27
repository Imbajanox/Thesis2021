from sklearn.metrics.pairwise import cosine_similarity

def similarWords (embeddingSpace, word, number=10):
    wordRepresentation = embeddingSpace[word].reshape(1,-1)
    words = []
    similarities = []
    for key,value in embeddingSpace.items():
        if key == word:
            continue
        words.append(key)
        similarities.append(cosine_similarity(wordRepresentation,value.reshape(1,-1)))
    sortedList = [list(x) for x in zip(*sorted(zip(words, similarities), key = lambda pair : pair[1], reverse=True))]
    return sortedList[0][:number]

def stability(word, sim1, sim2, same=False):
    if same and len(sim1) == 1:
        return len(sim1[0])

    sets1 = [set(a) for a in sim1]
    if not same:
        sets2 = [set(b) for b in sim2]
    else:
        sets2 = sets1

    avgOverlap = 0
    for i in range(len(sim1)):
        for j in range(len(sim2)):
            if not same or (same and i!=j):
                avgOverlap += len(sets1[i] & sets2[j])
    if same:
        avgOverlap /= (len(sim1)*len(sim2)-len(sim1))
    else:
        avgOverlap /= (len(sim1)*len(sim2))
    return avgOverlap

