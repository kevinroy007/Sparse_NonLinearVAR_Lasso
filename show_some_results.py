import matplotlib.pyplot as plt
import numpy as np
from NonlinearVAR import *

nlv_true = NonlinearVAR.from_pickle('nlv_true_0')
nlv_hat  = NonlinearVAR.from_pickle('nlv_hat_0')

m_A_true = np.linalg.norm(nlv_true.A, axis=2)
m_A_hat  = np.linalg.norm(nlv_hat.A,  axis=2)
plt.figure(); plt.matshow(m_A_true)
plt.figure(); plt.matshow(m_A_hat)

plt.show()
