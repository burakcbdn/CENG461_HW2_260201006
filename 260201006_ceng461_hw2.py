# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json # for saving generated data

# %%
# Read file

#from google.colab import drive
#drive.mount('/content/drive/',force_remount=True)

drivePath = "drive/MyDrive/IYTE/CENG461/corncob_lowercase.txt"
localPath = "corncob_lowercase.txt"

getFromLocal = True

txtFilePath = localPath if getFromLocal else drivePath

# %%
# creates cache for previously generated probabilities
probTables = {}

# %%
# add words to array
words = []
with open(txtFilePath, "r") as f:
  for line in f.readlines():
    words.append(line.replace('\n', '') + '*')


# %%
ms = "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,*"
msStates = ms.split(',')
markovChainStates = ms.split(',')

# {
#   a: {
#     0: 0,
#     total: 0,
#     a: 0,
#   ...
# probdata holds data for every state, their occurrences in first letter and occuring after specific letter
probData = {}

for state in markovChainStates:
  probData[state] = {
      '0': 0,
      'total': 0
  }


for word in words:
  for i in range(len(word)):
    letter = word[i]
    # increment total number of word
    probData[letter]['total'] = probData[letter]['total'] + 1
    if (i == 0):
      # save occurrence of letter in first position
      probData[letter]['0'] = probData[letter]['0'] + 1
      continue
    # save occurrence of letter after previous letter
    prevLetter = word[i - 1]

    probData[letter][prevLetter] = probData[letter].get(prevLetter, 0) + 1

probDF = pd.DataFrame.from_dict(probData).fillna(0.0)
probDF.sort_index(inplace=True)

# %%
def calculateAverageLength(words):
  total = 0

  for word in words:
    total += len(word) - 1 # minus 1 because of end of word character

  return total / len(words)

# %%
def calcPriorProb1(probTable, words, N):

  stateData = {}

  # initialize state data

  for state in msStates:
    stateData[state] = {
        'atloc': 0,
    }
  
  # get letter at position N
  for word in words:
    if (len(word) < N + 1):
      continue
    
    letterAtLoc = word[N]
    stateData[letterAtLoc]['atloc'] = stateData[letterAtLoc]['atloc'] + 1
  
  # calculate probabilitiy
  result = {}
  for state in msStates:
    result[state] = stateData[state]['atloc'] / probDF[state]['total']
  
  return result


# %%
# rescursive function to calculate joint conditional
def calculateConditional(probTable, word, state = '0'):
  
  if (len(word) == 1):
    return probTable[word][state]
  
  else:
    letter = word[-1]
    prevLetter = word[-2]

    val = calculateConditional(probTable, word[:-1])
    return probTable[letter][prevLetter] * val


# %%
def calcPriorProb2(probTable,N):

  stateData = {}

  # assuming word consists of same letters all, ex for state a and N = 5, our word will be "aaaaa"
  for state in msStates:
    if (state == '*'): continue # do not calculate for end of word
    word = state * N
    result = calculateConditional(probTable, word, state=state)
    stateData[state] = result

  return stateData


# %%
# calculate word probability using calculate conditional
def calcWordProb(probTable, word):
  return calculateConditional(probTable, word)

# %%
# generates prob table and saves it for order 
def generateProbTableForOrder(order):

  # check if prob table exists
  # if prob table exists use it
  # otherwise create prob table for this order

  table = probTables.get(order, 0)
  #table = 0
  #use existing prob table
  if (table != 0):
    return pd.DataFrame.from_dict(table)

  # generate all possible patterns for this order
  patterns = []

  def generatePatterns(n, letters, combination=""):
      if len(combination) == n:
          patterns.append(combination)
          return
      # Generate all possible combinations
      for letter in letters:
          generatePatterns(n, letters, combination + letter)

  generatePatterns(order, msStates[:-1])

  if (order == 1): patterns.append('0')

  # generate prob table for this order
  kOrderProbData = {}

  # initialize dictionary with msStates
  for state in msStates + ['total']:
    kOrderProbData[state] = {}

  # row data to store total occurence of pattern


  # calculate occurence of each letter after specific pattern
  for word in words:
    for i in range(len(word)):
      letter = word[i]
      if (order == 1 and i == 0):
        kOrderProbData[letter]['0'] = kOrderProbData[letter].get('0', 0) + 1

      if (i < order): continue
      # save occurrence of letter after specific patter 

      stateStr = word[i - order : i]


      kOrderProbData[letter][stateStr] = kOrderProbData[letter].get(stateStr, 0) + 1
      kOrderProbData['total'][stateStr] = kOrderProbData['total'].get(stateStr, 0) + 1

  if (order == 1):
    kOrderProbData['total']['0'] = len(words)

  df = pd.DataFrame.from_dict(kOrderProbData)

  df = df.set_index(df['total'].index)

  result = df.divide(df['total'], axis=0).fillna(0)
  result.drop(['total'], axis = 1, inplace=True)

  probTables[order] = result.to_dict()

  return result
    

# %%
# returns random letter from the given table and using previous pattern of order
def getKLetter(pTable, lookup, states):
  try:
    row = pTable.loc[lookup]
    arr = row.to_numpy().tolist()
    return np.random.choice(states, 1, p=arr)[0]
  except Exception as e: # if pattern not found just return random (this means probability was 0)
    return np.random.choice(states, 1)[0]



def generateWordWithOrder(order, pTable):

  word = ''


  # generate word until it ends with '*'
  while True:
    # select first letter using first order table
    if len(word) == 0:
      kProb = generateProbTableForOrder(1)
      letter = getKLetter(kProb, '0', msStates)
      word += letter
      continue
    # select second letter using first order table
    if len(word) == 1:
      kProb = generateProbTableForOrder(1)
      letter = getKLetter(kProb, word[0], msStates)
      if (letter == '*'): continue # if letter is '*' then continue 
      word += letter
      continue
    
    # select letter using smaller order table
    if (len(word) < order):
      kProb = generateProbTableForOrder(len(word))
      letter = getKLetter(kProb, word, msStates)
      word += letter
      if (letter == '*'): break
      continue
    # select letter using order table
    else:
      kProb = generateProbTableForOrder(order)
      letter = getKLetter(kProb, word[len(word) - order :len(word)], msStates)
      word += letter
      if (letter == '*'): break


  return word


# %%
def generateWordsWithOrder(order, M):
  # get initial prob table for order. If not exists then create it

  pTable = generateProbTableForOrder(order)

  kOrderWords = []

  for i in range(M):
    kOrderWord = generateWordWithOrder(order, pTable)
    kOrderWords.append(kOrderWord)

  return kOrderWords

# %%
def printWords(generatedWords, title=""):
  print("")
  print("Generated Words | " + title)
  for i in range(len(generatedWords)):
    print(f" {i + 1}. {generatedWords[i]}")
  print("")

# %% [markdown]
# ### Estimate P(L0) and P(LN | LN-1) and print it.

# %%
pl0 = generateProbTableForOrder(1)

print("")
print("Generated P(L0) and P(Ln | Ln-1) table")

#display(pl0)
print(pl0)
print("")
# %% [markdown]
# ### Calculate the average length of a word using the given list of words and print it.

# %%
average = calculateAverageLength(words)
print("")
print(f"Average length of a word: {average}")
print("")
# %% [markdown]
# ### Implement a function (calcPriorProb1) which takes the given list of words andN as input and returns P(LN). Plot the distributions for N=1,2,3,4,5 using bar plots.

# %%

# plot for N = 1,2,3,4,5 prior prob 1
for i in range(1,6):
  priorProb = calcPriorProb1(probTables[1], words, i)

  fig = plt.figure()
  plt.bar(msStates, priorProb.values())
  #plt.xlabel(f"N = {i}")
  plt.ylabel("Prior Probabilities 1")
  plt.title(f"Probability Distrubiton for N = {i}")

  plt.show()


# %% [markdown]
# ### Implement a function (calcPriorProb2) which takes P(L0),P(LN | LN-1) (estimated at Step 1) and N as input and returns P(LN). Plot the distributions for N=1,2,3,4,5 using bar plots.

# %%

# plot prior prob 2
for i in range(5):
  priorProb = calcPriorProb2(probTables[1], i + 1)

  fig = plt.figure()
  plt.bar(msStates[:-1], priorProb.values())
  plt.ylabel("Prior Probabilities 2")
  plt.title(f"Probability Distrubiton for N = {i + 1}")

  plt.show()


# %% [markdown]
# ### Calculate and print the probabilities for the following words: 
# 
# *   sad*
# *   exchange*
# *   antidisestablishmentarianism*
# *   qwerty*
# *   zzzz*
# *   ae*

# %%
wordsToCalculateProb = ["sad*", "exchange*", "antidisestablishmentarianism*", "qwerty*", "zzzz*", "ae*"]
print("")
for word in wordsToCalculateProb:
  probRes = calcWordProb(probTables[1], word)

  print(f"Probability of word '{word}': {probRes}")
print("")
# %% [markdown]
# #### Results
# *   sad* = 5.050673464418953e-05
# *   exchange* = 3.4260352437487854e-10
# *   antidisestablishmentarianism* = 1.6146855808797933e-31
# *   qwerty* = 0.0
# *   zzzz* = 2.250582734156605e-07
# *   ae* = 2.857933843042192e-05
# 

# %% [markdown]
# ### Generate 10 words
# 
# Sample generated words
# 
#  1. ess*
#  2. mpralin*
#  3. cees*
#  4. mig*
#  5. esoonttitss*
#  6. chonaleowoourypis*
#  7. sisced*
#  8. litmaveisesacrr*
#  9. sheel*
#  10. teraly*

# %%
generatedWords = generateWordsWithOrder(1, 10)


printWords(generatedWords, title="First order, 10 words")

# %% [markdown]
# ### By generating a synthetic dataset of size 100000, estimate the average length of a word and print it.

# %%
datasetSize = 100000

syntheticDataset = generateWordsWithOrder(1, datasetSize)
avg = calculateAverageLength(syntheticDataset)

print("")
print(f"Average Length of Word is '{avg}' for synthetic dataset of size '{datasetSize}'")
print("")

# %% [markdown]
# ### BONUS! increment the order of markov chain and genereate words

# %% [markdown]
# #### Order = 2
# 
# Sample generated words
#  1. cers*
#  2. gs*
#  3. hecrushiatermirch*
#  4. eddon*
#  5. futting*
#  6. pocuppets*
#  7. fula*
#  8. fech*
#  9. bey*
#  10. er*
# 

# %%
order2Words = generateWordsWithOrder(2, 10)

printWords(order2Words,  title="Second order, 10 words")

# %% [markdown]
# #### Order = 3
# 
# Sample generated words
# 
#  1. rises*
#  2. ts*
#  3. te*
#  4. ble*
#  5. dgetfullabordshipping*
#  6. supplying*
#  7. jectual*
#  8. fully*
#  9. cond*
#  10. cs*

# %%
order3Words = generateWordsWithOrder(3, 10)

printWords(order3Words, title="Third order, 10 words")

# %% [markdown]
# #### Order = 4
# 
# Sample generated words
# 
# 1. melando*
# 2. fflux*
# 3. majoritate*
# 4. ces*
# 5. proseconvergic*
# 6. cellings*
# 7. clasper*
# 8. ates*
# 9. clamating*
# 10. patrian*

# %%
order4Words = generateWordsWithOrder(4, 10)

printWords(order4Words,  title="Forth order, 10 words")

# %% [markdown]
# #### Order = 5
# 
# Sample generated words
# 
#  1. sed*
#  2. sly*
#  3. ives*
#  4. stairway*
#  5. etting*
#  6. by*
#  7. assination*
#  8. uckoos*
#  9. ng*
#  10. robes*

# %%
order5Words = generateWordsWithOrder(5, 10)

printWords(order5Words,  title="Fifth order, 10 words")

# %% [markdown]
# #### Order = 6
# 
# * try only if you have enough ram.
# 
# Sample generated words
# 
#  1. ara*
#  2. lly*
#  3. wholesalers*
#  4. bers*
#  5. stier*
#  6. umigations*
#  7. sy*
#  8. hers*
#  9. ation*
#  10. ocial*

# %%
order6Words = generateWordsWithOrder(6, 10)

printWords(order6Words,  title="Sixth order, 10 words")

# %% [markdown]
# #### Order = n
# 

# %%
n = 7

ordernWords = generateWordsWithOrder(7, 10)

printWords(ordernWords,  title="Nth order, 10 words")
