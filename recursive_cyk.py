
global G
G = ["S->AB", "S->XB", "T->AB", "T->XB", "X->AT", "A->a", "B->b"]
grid = {}
def parse(w, i, j, grid):
   if grid[i, j] :
      return grid[i, j]
   elif i == j:
      for rule in G:
        if rule.endswith(w[i]) and rule[0] not in grid[i, j]:    
             
                grid[i, j].append(rule[0])
   else:
      for k in range(i, j):
        for g1 in parse(w, i, k, grid):
    
                for g2 in parse(w, k+1, j, grid):

                    for rule in G:

                        if rule.endswith(g1+g2) and rule[0] not in grid[i, j]:

                            grid[i, j].append(rule[0])
   return grid[i, j]


def build_grid(St_Sym, word):
    S=St_Sym
    length= len(word)
    #grid=[[[] for x in range(length - y)] for y in range(length)]
    for x in range(0,length):
        for y in range(x, length):
            grid[x,y] = []
    
           
    comb = parse(w, 0, length - 1, grid)
    print (word, comb)

    if St_Sym in comb:
        print ("T")
        return True
    
    else:
        print ("F")
        return False


w1 = "aaaaabbbbb"
w2 = "aaaaaaaaabbbbbbbbb"


build_grid("S", w1)





