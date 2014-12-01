### Next
* implement batch learning from the collected feedback, instead of the perceptron-like updates
* filter the categories such as "... templates", "... stubs", "... articles", etc.

### Later

### Maybe
* implement space- and time-efficient storing of the category states

### Done
* query the user when too many steps have passed without feedback (just in case)
* implement memoization of the given feedback (not to ask the same question of the user twice)
* update the learning framework:
    * introduce the execution state object in order to reduce the number of parameters passed to the components of the learning algorithm
    * allow multiple querying conditions, restart and vector update rules
* implement the preference update when score(child) < score(parent)
* implement some form of querying rule based on the (low) confidence of the classifier
* clean up the update conditions and rules (with/without margins, etc.)
