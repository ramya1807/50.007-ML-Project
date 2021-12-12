# 50.007-ML-Project
## Project Overview
This project aims to design an efficient sequence labelling model for sentiment analysis using the Hidden Markov Model (HMM) as well as an the Viterbi Algorithm

## Execution
Provided below is a guide on how to run each part

### Part 1
Specify which dataset you wish to use when running each part like displayed below for each part:
```
python hmm_part1.py ES

python hmm_part1.py RU
```

### Part 2 
Run hmm_p2.py to output dev.p2.out
```
python3 "/50.007-ML-Project/output files/hmm_p2.py"
```
Change directory to ES or RU and run the following code to obtain the precision, recall and F scores.


```
python3 evalResult.py dev.out dev.p2.out
```
