import numpy as np


def EnumerateAll(mlist, m, n):
   # m < n
    if n == 0: return [[]] # end of recursion
    combinations = list() 
    for i in range(0,len(mlist)):
        num = mlist[i] 
        temp = mlist[i+1:] 
        if (m-i)>=n: #soalnya kalo m<n gak bs bikin apapun lgi
            for j in EnumerateAll(temp, m-1, n-1): 
                combinations.append([num]+j)
                # print(combinations)
        else : break
    return combinations

def SolveLP(A, b, G):
    #solve the linear prog problem

    # step 0: initialization
    minCOST = 0
    
    # step 1a: enumerate all combinations
    [m, n] = A.shape
    # print("solveLP mnA :", end='')
    # print(m,n,A)
    lst = EnumerateAll(np.arange(m), m, n)
    # print(lst)

    # step 1b: compute all the intersection points
    points = list()
    for idx in lst:
        Ai = A[idx, :]
        bi = b[idx]
        feasible = 1
        try: 
            xi = np.linalg.solve(Ai, bi)
        except np.linalg.LinAlgError:
            # Ai is singular or not square.
            feasible = 0
        
        # step 2: check the feasibility of the itersection point
        if feasible == 1:
            for i in range(m):
                if np.dot(A[i,:], xi) < b[i]:  # violate a constraints
                    feasible = 0
        if feasible == 1:            # only add the feasible point
            points.append(xi)
        
    # step 3: evaluate the G function for all intersection points
    values = list()
    for ptx in points:
        values.append(np.dot(G[0:n], ptx))
    
    # step 4: find the point with the smallest value as the result
    minCOST = min(values)
    minidx = values.index(minCOST)
    x = points[minidx]
    
    return x, minCOST
    
#-------------------------------#
# main program starts from here.#
#-------------------------------#
# Put all the coefficients of the constrains into a matrix A and a vector b

A = np.array([
            [210,180,220,60,400],
            [50,8,5,6,18],
            [2,12,40,1,30], 
            [6,6,2,5,6], 
            [1,10,4,1,15],
            [2,1,1,1,3],
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]
            ]) 
            
b = np.array([
            1800, 
            180, 
            140, 
            60, 
            30, 
            10, 
            0, 
            0, 
            0, 
            0, 
            0
            ])

G = np.array([30.0, 40.0, 10.0, 5.0, 20.0])


# solve this problem
[x, minCOST] = SolveLP(A, b, G)
print(x)
print(minCOST)

