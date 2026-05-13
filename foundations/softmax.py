import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        #pass

        # THOUGHT PROCESS
        # numerical stability - subtract max of z from all n within z
        # take e^n for stable n within z
        # sum these and divide each n within ans by it
        zmax = np.max(z)
        sum = 0
        ans = np.zeros(len(z))
        
        # compute numerator of softmax
        for n in range(len(z)):
            stable_n = z[n] - zmax
            en = np.exp(stable_n) 
            sum += en
            ans[n] = en 

        # compute denominator of softmax
        for n in range(len(ans)):
            ans[n] = ans[n] / sum
        return np.round(ans, 4)
            
        