def find_modified_max_argmax(L, f):
    V = [f(x) for x in L if type(x) == int]
    p = ()
    if V:
        m = max(V)
        p = m, V.index(m)
    
    return p
