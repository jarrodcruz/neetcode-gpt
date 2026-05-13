import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        #pass

        # THOUGHT PROCESS
        # take e^n for n within z
        # sum these and divide each e^n by it
        # numerical stability
        zmax = np.max(z)
        sum = 0
        ans = np.zeros(len(z))

        for n in range(len(z)):
            stable = z[n] - zmax
            en = np.exp(stable) 
            print(en)
            sum += en
            ans[n] = en 

        print(ans)
        for n in range(len(ans)):
            ans[n] = ans[n] / sum
        return np.round(ans, 4)
            
        