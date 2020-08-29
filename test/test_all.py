import unittest
import numpy as np
import fdasrsf as fs  

class TestFDASRSF(unittest.TestCase): 
  
    # Returns True or False.  
    def test_reparm(self):  
        M = 101
        q1 = np.sin(np.linspace(0,2*np.pi,M))
        timet = np.linspace(0,1,M)
        gam = fs.optimum_reparam(q1, timet, q1)       
        self.assertAlmostEqual(sum(gam-timet),0) 
  
if __name__ == '__main__': 
    unittest.main() 