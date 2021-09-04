import sys
import numpy as np
from collections import OrderedDict

class SVD:
    def __init__(self, array):
        self.array=array
        self.sigma=np.zeros(self.array.shape)
        self.U=[]
        self.V=np.zeros((self.array.shape[1], self.array.shape[1]))
        self.V_t=np.zeros((self.array.shape[1], self.array.shape[1]))
        self.sigma_values=[]

    def get_eigen_values(self, dict)->None:
        V=[]
        for key,value in dict.items():
            V.append(value)
        nparray=np.array(V)
        self.V=nparray.T
        self.V_t=nparray

    def find_sigma(self, values)->None:
        values=-np.sort(-values)#sorts values from greatest to least
        self.sigma_values=np.sqrt(values)
        row, col=np.diag_indices(self.sigma.shape[0])
        self.sigma[row, col]=np.array(values)
        self.sigma=np.sqrt(self.sigma)

    def find_V(self)->None:
        V=np.matmul(np.transpose(self.array), self.array)
        eigenvector=np.zeros(V.shape)
        eigen_values, eigenvector=np.linalg.eig(V)
        eigen_values=np.round(eigen_values, 3)
        eigenvector=np.round(eigenvector, 3)
        eigenvector=eigenvector.T
        eigen_dict=dict(zip(eigen_values, eigenvector))
        eigen_dict={k: v for k,v in sorted(eigen_dict.items(), reverse=True)}
        self.get_eigen_values(eigen_dict)
        
    def get_U_from_V(self)->None:
        U=[]
        for i,val in enumerate(self.sigma_values):
            x=(1/val)*(self.array)
            temp=np.resize(self.V_t[i], (self.array.shape[1],1))
            x=np.matmul(x, temp)
            U.append(x)
        self.U=np.concatenate([U[0], U[1]], axis=1)


    def find_U(self)->np.ndarray:
        U=np.matmul(self.array, np.transpose(self.array))
        eigen_values=np.linalg.eigvals(U)
        self.find_sigma(eigen_values)
        self.find_V()
        self.get_U_from_V()

        return self.U, self.sigma, self.V

    def SVD(self)->np.ndarray:
        U, Sigma, V=self.find_U()
        return U, Sigma, V

def main(arguments):
    """Main func."""
    array=np.array([[3,2,2],
                    [2,3,-2]])
    machine_learning_models=SVD(array)
    U, Sigma, V=machine_learning_models.SVD()

if __name__ == "__main__":
    main(sys.argv[1:])