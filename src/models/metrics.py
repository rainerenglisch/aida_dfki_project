import nltk.translate.bleu_score as ntbs
import rouge_score.rouge_scorer as rouge_scorer

def evaluate(c_true: dict, c_pred: dict, verbose:int = 0) -> tuple:
    """
    Calculate the performance of the model
    Metrics are: ROUGE-L and BLEU-1
    Arguments:
        c_true:  Dictionary containing all given captions per picture
                 key of the dictionary is the picture name
        c_pred:  Dictionary with one predicted caption per picture
                 key of the dictionary is the picture name
        verbose: Show insides regarding the verbose level
                 Levels:
                     0: show no insides
                     1: show ROUGE_L-Recall and BLEU-1-Precision lists
                     2: show internal data structures during run
                     3: show data inside loops
    Returns:
        ROUGE-L recall and BLEU-1 precision as float values within a tuple:
        Index:
            0: ROUGE-L recall
            1: BLEU-1 precision
    """
    # Init
    ROUGE_L_INDI = "rougeL"
    RECALL_INDEX = 1  # ROUGE-L score index
    WEIGTHS_1_GRAM = (1, 0, 0, 0)  # BLEU-1 config
    rouge_l_rec = []
    bleu_1_prec = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    crlf2 = lambda x: '\n' if x > 2 else ''

    # Calc scores
    for k in c_true.keys():
        references = c_true[k]
        if len(references) > 0:
            candidate = c_pred[k]  # one per picture

            # Calc ROUGE-L recall and BLEU-1 references
            rrec = []  # ROUGE-L recalls temp
            bref = []  # BLEU-1 references temp
            for reference in references:
                # Calc ROUGE-L recall
                score = scorer.score(reference, candidate)[ROUGE_L_INDI][RECALL_INDEX]
                rrec.append(score)
                if verbose >= 3:
                    print(f"Rouge-L-Recall: pred. caption '{candidate}'\ttrue caption '{reference:<30}':\t{score}")

                # BLEU-1 reference transformation
                bref.append(reference.split(" "))

            # ROUGE-L max value selection
            r_score = max(rrec)
            rouge_l_rec.append(r_score)
            if verbose >= 2:
                print(f"Rouge-L-Recalls pred. caption '{candidate}': {rrec} -> {r_score}{crlf2(verbose)}")

            # Calc BlEU-1 precision
            b_score = ntbs.sentence_bleu(bref, candidate.split(" "), weights=WEIGTHS_1_GRAM)
            bleu_1_prec.append(b_score)
            if verbose >= 3:
                print(f"List of BLEU-1 true captions: {bref}")
            if verbose >= 2:
                print(f"BLEU-1-Precision pred. caption: '{candidate}': {b_score}\n")

        else:
            print("Reference caption for image {k} wrong")

    # Prepare results
    rouge_l_rec_score = sum(rouge_l_rec)/len(rouge_l_rec)
    bleu_1_prec_score = sum(bleu_1_prec)/len(bleu_1_prec)

    if verbose == 1:
        print(f"ROUGE-L-Recalls:\n{rouge_l_rec} -> {rouge_l_rec_score}")
        print(f"\nBLEU-1-Precisions:\n{bleu_1_prec} -> {bleu_1_prec_score}")

    return rouge_l_rec_score, bleu_1_prec_score


# Test it
references = dict({
    "picture1.jpg": ["police killed the gunman", "police kills the gunman", "police has killed the gunman"], 
    "picture2.jpg": ["girl is playing with a doll", "little girl holds a doll", "blond girl is playing"],
    "picture3.jpg": ["car is standing on a lot", "car before a building ", "car is parking near a house"]
})
candidates = dict({
    "picture1.jpg": "police kill the gunman",
    "picture2.jpg": "girl is playing",
    "picture3.jpg": "near a building car is waiting"
})
assert evaluate(references, candidates, verbose=0) == (0.6666666666666666, 0.7666215479690409)