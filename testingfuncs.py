import scipy.sparse as spr
import numpy as np
from scipy.sparse.linalg import inv

def kronecker_product(A, B):

    return spr.kron(A, B.transpose())

def dot_product(A, B):

    return A.dot(B)

def pre_kronecker(matrix):

    matrix_dim = spr.csr_matrix.get_shape(matrix)[0]

    identity = spr.identity(matrix_dim)

    return spr.kron(matrix,identity)


def post_kronecker(matrix):

    matrix_dim = spr.csr_matrix.get_shape(matrix)[0]

    identity = spr.identity(matrix_dim)
    
    return spr.kron(identity, matrix.transpose())

def genLiouvillian(operator, operator_cross):

    term1 = 2*kronecker_product(operator,operator_cross)
    term2 = pre_kronecker(dot_product(operator_cross,operator))
    term3 = post_kronecker(dot_product(operator_cross,operator))

    #print(term3)


    Liouvillian = term1 - term2 - term3

    return Liouvillian

## Dimensions and Idenntities

dim_cavity = 35
dim_system = 2

identity_cavity= spr.identity(dim_cavity)
identity_system= spr.identity(dim_system)

## Creation and Annihalation Operators

diag = [ np.sqrt(element) for element in list(range(1,dim_cavity))]
                        
a_red = spr.diags(diag,1)
a_cross_red = spr.diags(diag,-1)


a = spr.kron(identity_system,a_red)
a_cross = spr.kron(identity_system,a_cross_red)


## 2LS operators

sig_red = spr.csr_matrix(np.array([[0, 1], [0, 0]]))
sig_cross_red = spr.csr_matrix(np.array([[0, 0], [1, 0]]))


sig = spr.kron(sig_red, identity_cavity)
sig_cross = spr.kron(sig_cross_red, identity_cavity)


### Hamiltonian Components

omega_s = 1
omega_a = 1

g = 1

H_2LS = omega_s * dot_product(sig_cross, sig)

H_cav = omega_a * dot_product(a_cross  , a  )

H_int = g * (dot_product(a_cross, sig) + dot_product(a, sig_cross))


### Build full Hamiltonian commutator

preH  = pre_kronecker(H_2LS)  + pre_kronecker(H_cav)  + pre_kronecker(H_int)

postH = post_kronecker(H_2LS) + post_kronecker(H_cav) + post_kronecker(H_int)

H_commutator = preH - postH


### Liouvillian Superoperator

gamma_se = 1000
spont_emission = genLiouvillian(sig,sig_cross)

gamma_ip = 999
incoherent_pump = genLiouvillian(a, a_cross)

sum_liouvillians = gamma_se/2 * spont_emission + gamma_ip/2 * incoherent_pump

Liouvillian_Superoperator = spr.lil_matrix(-(1j)*H_commutator + sum_liouvillians) #change format 

### Rho and inverse

h = int(np.sqrt(spr.csr_matrix.get_shape(Liouvillian_Superoperator)[0]))

rho_trace = np.zeros((h**2))
c = np.zeros((h**2,1))
c[0] = 1

c = spr.csr_matrix(c)

### Replace first row of L with Tr(rho) condition

for i in range(0,len(rho_trace)):

    if i % (h+1) ==0:
        rho_trace[i] = 1

    Liouvillian_Superoperator[0,i] = rho_trace[i]


## Calculate rho_ss

Liouvillian_Superoperator = spr.csc_matrix(Liouvillian_Superoperator)

M = inv(Liouvillian_Superoperator)

rho_ss = dot_product(M,c).toarray()

rho = np.reshape(rho_ss, (h,h))

print('h = ' + str(h))

print(rho)





