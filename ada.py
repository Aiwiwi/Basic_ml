import os
results = 'D:/BSU-ML/RNN_test/results.txt' 

temp = ''
with open(results, 'r', encoding='utf-8') as rf:
    for line in rf:
        for s in line:
            temp += s
            
with open('D:/BSU-ML/new.txt', 'w', encoding='utf-8') as rf:
    rf.write(temp)
    
    