def col2zero(arr, column):

    for col in range(len(arr)):
        arr[col][column] = 0

    return arr

def row2zero(arr, row_num):

    row_length = len(arr[row_num])
    for idx in range(row_length):
        arr[row_num][idx] = 0

    return arr

def remove_duplicates(arr):
    
    tokens = [arr]

    new_tokens = []
    s = " "
    for phrases in tokens:
        new_phrases = []
        phrases = [phrase.split() for phrase in phrases]
        for i in range(len(phrases)):
            phrase = phrases[i]
            if all([len(set(phrase).difference(phrases[j])) > 0 or i == j for j in range(len(phrases))]) :
                new_phrases.append(phrase)
        
        new_phrases = [s.join(phrase) for phrase in new_phrases]
        new_tokens.append(new_phrases)

    try:
        if len(new_tokens[0]) == 0:
            arr.sort(reverse=True)
            return [arr[-1].strip()]
    except:
        pass

    return new_tokens[0]