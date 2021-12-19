
import tensornetwork as tn
import numpy as np
from tensornetwork.network_components import connect

# Structure:
# 0   1   2
#  \ / \ /
#   3   4
#  / \ / \
# 5   6   7
#  \ / \ /
#   8   9
#  / \ / \
# 10  11  12

Link=[[[3,6],[4,7],[8,11],[9,12]],[[3,5],[4,6],[8,10],[9,11]],
[[0,3],[1,4],[5,8],[6,9]],[[1,3],[2,4],[6,8],[7,9]]]


def Operators():
    X=1/np.sqrt(2)*np.array([[1,-1j],[-1j,1]])
    Y=1/np.sqrt(2)*np.array([[1,-1],[1,1]])
    W=1/np.sqrt(2)*np.array([[1,-np.sqrt(1j)],[np.sqrt(-1j),1]])
    return X,Y,W

def fSim(theta,phi):
    A=np.array([[1,0,0,0],[0,np.cos(theta),-1j*np.sin(theta),0],
    [0,-1j*np.sin(theta),np.cos(theta),0],[0,0,0,np.exp(-1j*phi)]])
    A=np.reshape(A,(2,2,2,2))
    return A

def getranC(D):
    # in this section, we generate the random circuit that we are going to simulate
    N=13
    SGate=np.zeros((D+1,N,2,2),dtype=complex)
    TGate=np.zeros((D,4,2,2,2,2),dtype=complex)
    X,Y,W=Operators()
    for d in range(D+1):
        for n in range(N):
            r=np.random.randint(0,3)
            if r==0:
                SGate[d][n]=X
            elif r==1:
                SGate[d][n]=Y
            else:
                SGate[d][n]=W
    for d in range(D):
        for k in range(4):
            theta=np.random.uniform(0,2*np.pi)
            phi=np.random.uniform(0,2*np.pi)
            TGate[d][k]=fSim(theta,phi)
    return SGate,TGate


def apply_gate(qubitE, gate,opqubit):
    op=tn.Node(gate)
    for i,bit in enumerate(opqubit):
        tn.connect(qubitE[bit],op[i])
        qubitE[bit]=op[i+len(opqubit)]

def svd_unitary(gate):
    A=np.zeros((2,2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    A[i][j][k][l]=gate[i][k][j][l]
    A=np.reshape(A,(4,4))
    U,S,V=np.linalg.svd(A)
    S=np.diag(S)
    Lm=np.dot(U,np.sqrt(S))
    Rm=np.dot(np.sqrt(S),V)
    


def Contract(SGate,TGate,cut,N,D):
    state=np.array([[1.0+0.0j,0.0+0.0j],[0.0+0.0j,1.0+0.0j]])
    depth=cut[0]
    num=cut[1]
    crosscut=0
    BigheadNode=[]
    for d in range(depth,D):
        for w in Link[d]:
            if w[0]<num and w[1]>=num:
                crosscut=1
    print(crosscut)
    if crosscut==0:
        with tn.NodeCollection(BigheadNode):
            StateNode=[tn.Node(state[0]) for _ in range(N)]
            qubits=[node[0] for node in StateNode]
            for d in range(D):
                if d<depth:
                    for n in range(N):
                        apply_gate(qubits,SGate[d][n],[n])
                    for i,w in enumerate(Link[d]):
                        apply_gate(qubits,TGate[d][i],w)
                else:
                    for n in range(num,N):
                        apply_gate(qubits,SGate[d][n],[n])
                    for i,w in enumerate(Link[d]):
                        if w[0]>=num:
                            apply_gate(qubits,TGate[d][i],w)
            for n in range(num,N):
                apply_gate(qubits,SGate[D][n],[n])
            EndNode=[tn.Node(state[0]) for _ in range(num,N)]
            Ebits=[Enode[0] for Enode in EndNode]
            for n in range(num,N):
                tn.connect(qubits[n],Ebits[n-num])
        bigvec=tn.contractors.auto(BigheadNode,output_edge_order=qubits[:num])
        bigvec=bigvec.tensor
        #print(np.shape(vec1.tensor))
        for z in range(2**num):
            vec1=tn.Node(bigvec)
            strz=format(z,'b').zfill(num)
            SmallheadNode=[]
            with tn.NodeCollection(SmallheadNode):
                StateNode=[]
                qubits=[]
                con=[]
                for i in range(num):
                    s=int(strz[i])
                    Nnode=tn.Node(state[s])
                    StateNode.append(Nnode)
                for n in range(num):
                    op=tn.Node(SGate[depth][n])
                    qubits.append(op[1])
                    con.append(op[0])
                for d in range(depth,D):
                    for i,w in enumerate(Link[d]):
                        if w[1]<num:
                            apply_gate(qubits,TGate[d][i],w)
                    for n in range(num):
                        apply_gate(qubits,SGate[d+1][n],[n])
                for n in range(num):
                    tn.connect(qubits[n],StateNode[n][0])
            vec2=tn.contractors.auto(SmallheadNode,output_edge_order=con)
            vec2=tn.Node(vec2.tensor)
            # print(np.shape(vec1.tensor))
            # print(np.shape(vec2.tensor))
            for i in range(num):
                tn.connect(vec1[i],vec2[i])
            result=vec1@vec2
            
            print((strz,result.tensor))



def FContract(SGate,TGate,N,D):
    Qnode=[]
    state=np.array([[1.0+0.0j,0.0+0.0j],[0.0+0.0j,1.0+0.0j]])
    with tn.NodeCollection(Qnode):
        StateNode=[tn.Node(state[0]) for _ in range(N)]
        qubits=[node[0] for node in StateNode]
        for d in range(D):
            for n in range(N):
                apply_gate(qubits,SGate[d][n],[n])
            for i,w in enumerate(Link[d]):
                apply_gate(qubits,TGate[d][i],w)
        for n in range(N):
            apply_gate(qubits,SGate[D][n],[n])
        EndNode=[tn.Node(state[0]) for _ in range(N)]
        Ebits=[Enode[0] for Enode in EndNode]
        for i in range(N):
            tn.connect(qubits[i],Ebits[i])
    result=tn.contractors.auto(Qnode)
    print(result.tensor)

D=4
N=13
cut=(2,5)
# SGate,TGate=getranC(D)
# Contract(SGate,TGate,cut,N,D)
# FContract(SGate,TGate,N,D)

A=np.array([[1,2,3,4],[1,2,3,4]])
B=np.reshape(A,(2,2,2))
print(B)