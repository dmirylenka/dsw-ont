### Next
* implement batch learning from the collected feedback, instead of the perceptron-like updates
* implement memoization of the given feedback (not to ask the same question of the user twice)
* update the learning framework:
    * allow multiple querying conditions, restart and vector update rules


### Later

### Maybe
* implement space- and time-efficient storing of the category states

### Done
* update the learning framework:
    * introduce the execution state object in order to reduce the number of parameters passed to the components of the learning algorithm
* implement the preference update when score(child) < score(parent)
* implement some form of querying rule based on the (low) confidence of the classifier
* clean up the update conditions and rules (with/without margins, etc.)
