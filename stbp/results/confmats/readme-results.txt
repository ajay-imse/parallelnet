stbp-ibmdataset-results
-

network = 64x64-20c5-2p-50c5-2p-200fc-10o
1. train and test one whole dset, acc=99.59%
2. splitnet with static delay, acc= 99.91%

3. k4fold test stbp, acc=95.9+-5.464 [87.8689, 97.3770, 98.3607, 100]
4. k=4 kfold test with delay acc=96.3115+-4.1071 [92.1311,100,93.4426,99.6721]
-


network with delay
learning as is
i.e. part of network output is buffered to next input
5 ms delay - accuracy 91.12%
10 ms delay - accuracy 96.31%
15 ms delay - accuracy 89.9%
20 ms delay - accuracy 85.4%
-
learning and weight replication. 
i.e. weights are copied to the neurons with delay from weights of neurons without delay
weights are replicated for Conv layer 1 only
10 ms delay - accuracy 92.5%

weights are replicated to C1 and C2
10 ms delay - accuracy 81.1%
