### Next
* introduce monitoring/stopping based on the performance on the nodes visited so far
* implement the stopping criterion
* figure out the interplay between C and gamma
* on negative feedback, figure out the first irrelevant parent
* filter the categories such as "templated, stubs, etc."

### Later

### Maybe
* implement space- and time-efficient storing of the category states

### Done
* make too-long-without-feedback start counting only if the graph has grown bigger than previously
* present the query node with the path to the root
* distinguish automatic feedback points from user-input ones
* restart if too_long_without_feedback only if we had negative feedback
* implement batch learning from the collected feedback, instead of the perceptron-like updates
* query the user when too many steps have passed without feedback (just in case)
* implement memoization of the given feedback (not to ask the same question of the user twice)
* update the learning framework:
    * introduce the execution state object in order to reduce the number of parameters passed to the components of the learning algorithm
    * allow multiple querying conditions, restart and vector update rules
* implement the preference update when score(child) < score(parent)
* implement some form of querying rule based on the (low) confidence of the classifier
* clean up the update conditions and rules (with/without margins, etc.)
