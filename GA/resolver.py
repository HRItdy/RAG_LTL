#True = true
#False = false
#N = Not
#A = And
#O = Or 
#G = Always
#E = Eventually
#X = Next
#U = Until


def progress(formula, assignment):
    '''
        Progression function to update the LTL instructions left to accomplish.

        formula: List[List[...]] | List[...] | str
        assignment: List[predicates] 
    '''

    # no operator case
    if type(formula) == str:
        if len(formula) == 1:
            return "True" if formula in assignment else "False"
        # already a bool str: { "True" | "False" }
        return formula

    assert len(formula) > 0, "Empty LTL formula."
    op = formula[0]

    # AND operator
    if(op == "A"):
        res1 = progress(formula[1], assignment)
        res2 = progress(formula[2], assignment)
        if res1 == "True" and res2 == "True":   return "True"
        if res1 == "False" or res2 == "False":  return "False"
        if res1 == "True":  return res2
        if res2 == "True":  return res1
        if res1 == res2:    return res1
        return ["A", res1, res2]
        
    # OR operator
    elif(op == "O"):
        res1 = progress(formula[1], assignment)
        res2 = progress(formula[2], assignment)
        if res1 == "True" or res2 == "True":    return "True"
        if res1 == "False" and res2 == "False": return "False"
        if res1 == "False": return res2
        if res2 == "False": return res1
        if res1 == res2:    return res1
        return ["O", res1, res2]
    
    # NOT operator
    elif(op == "N"):
        res = progress(formula[1], assignment)
        if res == "True":   return "False"
        if res == "False":  return "True"
        return ["N", res]

    # ALWAYS operator
    elif(op == "G"):
        res = progress(formula[1], assignment)
        if res == "False":  return "False"
        if res == "True":   return formula
        return ["G", res]

    # EVENTUALLY operator
    elif(op == "E"):
        res = progress(formula[1], assignment)
        if res == "True":   return "True"
        if res == "False":  return formula
        return ["E", res]

    # NEXT operator
    elif(op == "X"):
        res = progress(formula[1], assignment)
        return res
    
    # UNTIL operator
    # elif(op == "U"):
    #     res1 = progress(formula[1], assignment)
    #     res2 = progress(formula[2], assignment)
    #     if res2 == "True":  return "True"
    #     if res1 == "False": return res2
    #     if res2 == "False":
    #         if res1 == "True":  return formula
    #         return ["A", res1, formula]
    #     if res1 == "True":
    #         if res2 == "False": return formula
    #         return ["O", res2, formula]
    #     return ["O", res2, ["A", res1, formula]]
    
    elif(op == "U"):
        res1 = progress(formula[1], assignment)
        res2 = progress(formula[2], assignment)
        if res2 == "True":  return "True"
        if res1 == "False": return res2
        if res2 == "False":
            if res1 == "True":  return formula
            #return ["A", res1, formula]
            return formula # Temporally set as original formula to avoid propagation of task
        if res1 == "True":
            if res2 == "False": return formula
            # return ["O", res2, formula]
            return formula # same purpose
        #return ["O", res2, ["A", res1, formula]]
        return formula # same purpose

    else:
        # unknown operator found
        raise NotImplementedError


def is_accomplished(formula):
    '''
        Helper function to check for goal satisfying.
        Return True in case of only safety constraints left, False otherwise.
    '''

    # base cases
    if type(formula) == str:    return False
    elif formula[0] == 'G':     return True
    
    # recursive calls
    return all([is_accomplished(formula[i]) for i in range(1, len(formula))])



if __name__ == "__main__":

    assignment = ['r']
    # formula = ["E", ["A", "b", ["E", "r"]]]
    formula = ['A', ['G', ['N', 'b']], ['E', 'r']]

    print(formula)
    print(is_accomplished(formula))

    result = progress(formula, assignment)
    print(result)
    print(is_accomplished(result))

