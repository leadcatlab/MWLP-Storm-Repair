
def permutation(list):
    if len(list) == 0:
        return []

    if len(list) == 1:
        return [list]

 
    l = []
 
    #for our purposes, we can simply generate permuations starting from first node
    for i in range(1, len(list)):
       curr = list[i]
 
       rem = list[:i] + list[i+1:]
 
       for p in permutation(rem):
           l.append(p + [curr])
    return l
 
