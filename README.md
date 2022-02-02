# Decision Tree CART and Naive Bayes

Application of Machine Learning methods (CART and Naive Bayes) in the context of a study carried out by Columbia University where data on participants in experimental speed dating events were collected.

During the experiment, university students had quick "first encounters" lasting 4 minutes with all other participants. At the end of these 4 minutes, participants filled out a questionnaire on whether they would like to see their partner again. However, the dataset contained other information about the participants, namely: id (participant's identifier number), partner (peer's identifier number), age (participant's age), age_o (peer's age), goal (which the participant's objective), date (frequency with which you leave for meetings), go_out (frequency with which you leave), int_corr (correlation of interest), duration (opinion regarding the duration of the meeting), met (if you already knew the pair), like (if you liked the pair), prob (probability of the pair like you) and match, which is the objective class for our study. 

First you need to get the libraries: pandas, scikit and graphvitz
```
pip install pandas

pip install -U scikit-learn

pip install graphviz
```

###########################################################################################################

Command to run the Decision Tree:
```
python '.\Decision Tree.py'
```

Command to run the Naive Bayes:
```
python '.\Naive Bayes.py'
```
