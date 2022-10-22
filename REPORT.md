REPORT:
- unfortunately, my computer is far too slow and limited in compute to run the
  model at anywhere near full capacity so my hyperparameter choices are essentially
  just to allow me to run the model. I cannot really asses the performance of the
  model(s) on either in vitro or (especially) in vivo tests. However, I will discuss
  the effects of each hyperparameter, as well as each eval task:
  HYPER PARAMS/IMPLEMENTATION:
  - embedding size: within the skipgram model, this serves as the size of the word
    vector itself. The dimensionality here serves to capture the information of the
    word and its context. A larger size will allow for more information, but the goal
    here is essentially to compress the co-occurence of the word with context words
    (or vice verse for CBOW) so something relatively small is the goal. I chose 128
    somewhat arbitrarily.
  - vocab size: This can increase model training time as more words must be learned,
    and more importantly it makes the size of the model itself bigger. Since word
    frequency follows a rapidly decaying distribution, it does not require too large
    of a vocab to capture 90% of all words in the books. 3000 is a good choice, but
    I have seen 10k as the standard in many papers
  - context window size: increasing this should improve the model's semantic accuracy,
    while decreasing it will improve syntactic performance. I chose 5 as a good medium
    between the two
  - batch size: increasing this will improve performance at the cost of efficiency (if
    too large). I would choose the largest batch that would fit on my computer since
    that would be an optimum between performance and efficiency.
  - num epochs: This one is again arbitrary. Keep training until the model converges,
    but the training/test errors don't start to diverge
  - learning rate: This affects the speed of convergence, but an LR that is too big won't
    converge properly. Chosen arbitrarily.
  - choosing the top-k predictions for accuracy: from the model's predicted logits, I just 
    take the top k preds (k being 2*context window size). This is easy (that is why I did 
    it), but selecting all indicies above a certain threshold would also be an interesting
    choice. This latter method could allow for the model to choose less than the full
    context window number of tokens, thus implying repeated tokens. This may give a
    more correct understanding of the model's accuracy.
  - accuracy calculation: I calculated accuracy as size of intersection of target & pred
    divided by size of target. This might be inaccurate in the case of repeated words 
    (ie. intersection is only 5, but the total number of unique words in the target is 
    also 5). Another option would've been intersection over union, which would've been
    more indicative of true performance. However, I think since I am only choosing the
    top k logits (as described above), this shouldn't be a major issue. If I wasn't
    choosing the top k logits, it would be possible for the model to predict 1 for all
    tokens, and thus have an accuracy of 100%

  EVALUATION TASKS:
  - The in-vitro task is the loss function/accuracy calculation for the language modeling
    task itself
    - Accuracy may be incorrectly over-estimating model performance, which is described in
      "accuracy calculation" above
  - The in-vivo evaluation tasks are split into two main categories: semantic & syntactic
  - The semantic tasks analyze the embeddings' abilities to capture the meaning of words.
    - This means that the relationships in the analogies used are based on the words'
      meanings with respect to each other. For example, capitals is clearly testing to see
      if the embeddings can capture the relationship between a country and its capital. Or
      for things like attributeof/instanceof, the embeddings should be able to capture the
      meaning of something being an attribute/instance of something else.
  - The syntactic tasks instead analyze the embeddings' abilities to capture the relationships
    of words to each other with respect to syntax.
    - This means that the analogies have relationships based on syntaxes of sentences. For
      example, plural_noun tests to see if the embeddings capture the difference between a
      singular and plural noun (as oppsoed to verb which is tested seperately).
    - Superlative, instead, captures a words' syntactic meaning in terms of extremes. This
      task measures if the embeddings capture converting a word to its extreme.
  - The metrics calculated for in vivo are exact, MRR, and MR
    - Exact guages how many tasks were guessed exactly right (correct word was first rank)
      - This is not a great charactarization of the model's performance since the correct token
        may always appear second or third. These instances would get a score of 0, thus under-estimating
        performance
    - MR (mean rank) is an aggregate of the rank that the correct word had in the predictions
      (lower is better)
    - MRR (mean recipricol rank) is the inverse of MR, and so higher is better
    
EXTRA CREDIT:

I implemented CBOW after Skip Gram. I cannot analyze their relative performance because my computer
will take hours/days to train on the full dataset. To train CBOW, run the same command as usual, but
on the train_cbow.py file!