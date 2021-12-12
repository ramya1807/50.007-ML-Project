import sys
import os
import math
import copy

root_dir = "./"

# count number of time - transition
def count_transition(file):
    with open(file, "r", encoding="cp437", errors='ignore') as f:
        lines = f.readlines()
    
    state = "START" #first state
    transition_track = {}
    
    for line in lines:
        line_split = line.strip()  
        line_split = line_split.rsplit(" ")  
        
        # case 1: word line
        if len(line_split) == 2:
            token = line_split[0]
            state_u = line_split[1]
            if state not in transition_track:
                state_dict = {}
            else:
                state_dict = transition_track[state]
            if state_u in state_dict:
                state_dict[state_u] += 1
            else:
                state_dict[state_u] = 1

            transition_track[state] = state_dict
            state = state_u
    
    return transition_track


#transition parameters
def transition_parameters(transition_tracker, state, state_u):
    if state not in transition_tracker:
        fraction = 0
        
    else:
        state_dict = transition_tracker[state]
        numerator = state_dict.get(state_u, 0)
        denominator = sum(state_dict.values())
        fraction = numerator / denominator
    
    return fraction

# count number of times - emission
def count_emission(file):
    with open(file, "r", encoding="cp437", errors='ignore') as f:
        lines = f.readlines()
        
    all_token = set()
    emission_tracker = {}
    
    for line in lines:
        split_line = line.strip()  
        split_line = split_line.rsplit(" ")  #remove excess

        if len(split_line) == 2:
            token = split_line[0]
            tag = split_line[1]        
            all_token.add(token)

            if tag in emission_tracker:
                current_emission = emission_tracker[tag]
            else:
                current_emission = {}

            if token in current_emission:
                current_emission[token] = current_emission[token] + 1
            else:
                current_emission[token] = 1
            emission_tracker[tag] = current_emission

    return all_token, emission_tracker



#emission parameters
def emission_parameters(emission_tracker, x, y, k = 1.0): #set k to 1 according to part 1
    state_dict = emission_tracker[y] 
    denom = sum(state_dict.values()) + k
    
    if x != "#UNK#":
        num = state_dict[x]
    else: 
        num = k
    return num / denom

#Viterbi
def viterbi(emission_dict, transition_dict, observations, sentence):
    n = len(sentence)
    smallest = -9999

    #set up state/token and score
    states = list(transition_dict.keys())
    states.remove("START")
    scores = {}

    #states --> start to first state
    scores[0] = {}
    for state_v in states:
        transition_prob = transition_parameters(transition_dict, "START", state_v)
        if transition_prob != 0:
            transition = math.log(transition_prob)
        else:
            transition = smallest
        
        # if the word does not exist, assign special token
        if sentence[0] not in observations:
            token = "#UNK#"
        else:
            token = sentence[0]

        if token in emission_dict[state_v]: 
            emission_prob = emission_parameters(emission_dict, token, state_v)
            emission = math.log(emission_prob)
        elif token == "#UNK#":
            emission_prob = emission_parameters(emission_dict, token, state_v)
            emission = math.log(emission_prob)
        else:
            emission = smallest
        
        first = transition + emission
        scores[0][state_v] = ("START", first)
    
    
    #states --> first onwards
    for i in range(1, n):
        scores[i] = {}
        for state_v in states:
            cal_max = []
            for state_u in states:
                transition_prob = transition_parameters(transition_dict, state_u, state_v)
                if transition_prob != 0:
                    transition = math.log(transition_prob)
                else:
                    transition = smallest
                if sentence[i] not in observations:
                    token = "#UNK#"
                else:
                    token = sentence[i]

                if token in emission_dict[state_v]: 
                    emission_prob = emission_parameters(emission_dict, token, state_v)
                    emission = math.log(emission_prob)
                elif token == "#UNK#":
                    emission_prob = emission_parameters(emission_dict, token, state_v)
                    emission = math.log(emission_prob)
                else:
                    emission = smallest
                
                n_score = scores[i-1][state_u][1] + transition + emission
                cal_max.append(n_score)
    
            # argmax for y
            final_score = max(cal_max)
            state_score = states[cal_max.index(final_score)]
            scores[i][state_v] = (state_score, final_score)
    
    #states until stop
    scores[n] = {}
    stop_max = []
    for state_u in states:
        transition_prob = transition_parameters(transition_dict, state_u, "STOP")
        if transition_prob != 0:
            transition = math.log(transition_prob)
        else:
            transition = smallest

        if token in emission_dict[state_v]: 
            emission_prob = emission_parameters(emission_dict, token, state_v)
            emission = math.log(emission_prob)
        elif token == "#UNK#":
            emission_prob = emission_parameters(emission_dict, token, state_v)
            emission = math.log(emission_prob)
        else:
            emission = smallest
        
        stopscore = scores[n-1][state_u][1] + transition + emission
        stop_max.append(stopscore)
    
    #agmax of y
    stop = max(stop_max)
    state_score = states[stop_max.index(stop)]

    scores[n] = (state_score, stop)

    #go backwards to find token in sequence
    sequence = ["STOP"]
    final = scores[n][0]
    sequence.insert(0, final)
    
    for k in range(n-1, -1, -1):
        final = scores[k][final][0] 
        sequence.insert(0, final)
    return sequence


if __name__ == '__main__':
    datasets = ["ES", "RU"]

    for i in datasets:
        train = root_dir + "{folder}/train".format(folder = i)
        evaluation = root_dir + "{folder}/dev.in".format(folder = i)

        all_tokens, track_emission = count_emission(train)
        track_transition = count_transition(train)

        with open(evaluation, "r", encoding="cp437", errors='ignore') as f:
            lines = f.readlines()

        sentence = []
        predict = []
        print(i)


        for line in lines:        
            if line != "\n":
                line = line.strip()
                sentence.append(line)
            else:
                sentence_predict = viterbi(track_emission, track_transition, all_tokens, sentence)
                sentence_predict.remove("START")
                sentence_predict.remove("STOP")
                predict = predict + sentence_predict
                predict = predict + ["\n"] #add in order
                sentence = []
        



        # create dev.p2.out
        with open(root_dir + "{folder}/dev.p2.out".format(folder = i), "w", encoding="cp437", errors='ignore') as g:
            for j in range(len(lines)):
                word = lines[j].strip()
                if word != "\n":
                    tag = predict[j]
                    if(tag != "\n"):
                        g.write(word + " " + tag)
                        g.write("\n")
                    else:
                        g.write("\n")

    print("Done")