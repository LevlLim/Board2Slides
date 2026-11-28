A=["bookkeeper","parallel","access","food","weights","ghosts"]; B=["book keeper","letter","gradient"]; C=["earth","generalization"]
D=["The model should preserve repeated letters in words such as Aiken and Xavier.","The white board contains a mixture of short words, long phrases and numbers","A careful crop keeps every hand written character inside the recognition region"]
E=["Different writers use different spacing, pressure, height on the same board.","Please leave enough space between lines, so the detector does not merge neighbouring sentences"]
def refs(n):
    n=int(n)
    if 5095<=n<=5108: return "A",(A[:4] if n==5097 else A)
    if 5109<=n<=5112: return "B",B
    if 5113<=n<=5115: return "C",C
    if 5116<=n<=5124: return "D",(D[:2] if 5119<=n<=5121 else D)
    return "E",E

def ignored(n):
    n=int(n)
    if n==5097: return A[4:]
    if 5119<=n<=5121: return D[2:]
    return []
