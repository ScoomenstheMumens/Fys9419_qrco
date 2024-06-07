import numpy as np
from qiskit.algorithms import MinimumEigensolver, VQEResult
from qiskit.quantum_info import SparsePauliOp
import cmath
from util import cost_string
import util
import math
import ansatz_circ 
import mthree
from IPython.display import clear_output
class QRAO_encoding_VQE(MinimumEigensolver):
    
    def __init__(self,estimator,sampler, circuit, optimizer,graph,min,shots=None,initial_parameters=None,callback=None):
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._obs=util.operator_vertex_pauli(graph)
        self._num_qubits=self._circuit[0].num_qubits
        self._graph=graph
        self._sampler=sampler
        self._ham=util.edge_pauli(graph)
        print(self._obs)
        print(self._ham)
        
    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            qc=self._circuit[0].bind_parameters(x)
            qc.remove_final_measurements()
            job = self._estimator.run(circuits=qc, observables=self._ham,shots=self._shots)
            H=job.result().values
            if self._callback is not None:

                self._callback([H[0],[x]])
                print(H)
            return H[0]
            
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.ones(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)

        exp_vertex=[]
        string=[]
        qc=self._circuit[0].bind_parameters(res.x)
        qc.remove_final_measurements()
        for op in self._obs:
            print(op[0])
            job = self._estimator.run(circuits=qc, observables=op[0],shots=self._shots)
            exp_vertex.append(job.result().values)

        print(res.x)
        print(exp_vertex)
        for exp in exp_vertex:
            if np.sign(exp)<0:
                string+='0'
            else:
                string+='1'

        print(string)
        print(util.cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result

class QRAO_encoding_VQE(MinimumEigensolver):
    
    def __init__(self,estimator,sampler, circuit, optimizer,graph,min,shots=None,initial_parameters=None,callback=None):
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._obs=util.operator_vertex_pauli(graph)
        self._num_qubits=self._circuit[0].num_qubits
        self._graph=graph
        self._sampler=sampler
        self._ham=util.edge_pauli(graph)
        print(self._obs)
        print(self._ham)
        
    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            H=0
            exp_vertex=[]
            string=[]
            qc=self._circuit[0].bind_parameters(x)
            qc.remove_final_measurements()
            for op in self._ham:
                #print(op[0])
                job = self._estimator.run(circuits=qc, observables=op[0],shots=self._shots)
                exp_vertex.append(job.result().values)
                #H-=np.real(op.coeffs)*(1-np.real(job.result().values)/np.real(op.coeffs))/2
                H+=np.real(job.result().values)/2
            if self._callback is not None:

                self._callback([H,[x],exp_vertex])
                print(H)
                print(exp_vertex)
            return H
            
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.ones(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)

        exp_vertex=[]
        string=[]
        qc=self._circuit[0].bind_parameters(res.x)
        qc.remove_final_measurements()
        for op in self._obs:
            print(op[0])
            job = self._estimator.run(circuits=qc, observables=op[0],shots=self._shots)
            exp_vertex.append(job.result().values)

        print(res.x)
        print(exp_vertex)
        for exp in exp_vertex:
            if np.sign(exp)<0:
                string+='0'
            else:
                string+='1'

        print(string)
        print(util.cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result


class PauliEncoding_VQE(MinimumEigensolver):
    
    def __init__(self,estimator, graph, circuit, optimizer,observables,num_qubits,min,alpha,beta,v,gamma,shots=None,initial_parameters=None,callback=None):
        self._graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._obs=observables
        self._alpha=alpha
        self._num_qubits=num_qubits
        self._beta=beta 
        self._gamma=gamma
        self._v=v

        
    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            qc=self._circuit.bind_parameters(x)
            job = self._estimator.run(circuits=[qc]*len(self._obs), parameters=x, observables=self._obs,shots=self._shots)
            H=0
            exps=[]
            for value in job.result().values:
                 exps.append(value)
            exps=np.array(exps)
            exps=exps/np.sum(exps)
            for edge in self._graph.edges():
                H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.tanh(self._alpha*exps[edge[0]-1])*np.tanh(exps[edge[1]-1]))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.cos((self._alpha*job.result().values[edge[0]-1]/2-1/2)*np.pi)*np.cos((self._alpha*job.result().values[edge[1]-1])/2-1/2)*np.pi)/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-job.result().values[edge[0]-1]*job.result().values[edge[1]-1])/2
            
            reg=0
            for i in range(0,len(self._obs)):
                reg+=1/self._num_qubits*(np.tanh(np.abs(self._alpha*exps[i])-self._gamma)**2)
            
            reg=self._beta*self._v*reg**2
            
            if self._callback is not None:

                self._callback([H,reg,x])
            H+=reg
            return H
            
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.ones(self._circuit.num_parameters)
            
        else:
            x0=self.initial_parameters
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        qc=self._circuit.bind_parameters(res.x)
        job = self._estimator.run(circuits=[qc]*len(self._obs), parameters=res.x, observables=self._obs,shots=self._shots)
        H=0
        exps=[]
        for value in job.result().values:
            exps.append(value)
        exps=np.array(exps)
        exps=exps/np.sum(exps)
        for i,edge in enumerate(self._graph.edges()):
            H-=self._graph[edge[0]][edge[1]].get('weight', None)*(1-np.sign(self._alpha*exps[edge[0]-1])*np.sign(self._alpha*exps[edge[1]-1]))/2

        string=[]
        for i in range(len(self._obs)):
            if np.sign(self._alpha*exps[i])<0:
                string+='0'
            else:
                string+='1'
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result


class Pauli_efficient_VQE(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,observables,num_qubits,min,alpha,beta,v,gamma,shots=None,initial_parameters=None,callback=None):
        self._graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._obs=observables
        self._alpha=alpha
        self._num_qubits=num_qubits
        self._beta=beta 
        self._gamma=gamma
        self._v=v
        op_x=[]
        op_y=[]
        op_z=[]    
        for op in self._obs:
            if 'X' in op:
                op=op.replace('X', 'Z')
                op_x.append(op)
            if 'Y' in op:
                op=op.replace('Y', 'Z')
                op_y.append(op)
            if 'Z' in op:
                op_z.append(op)
        self._obs=[op_x,op_y,op_z]
        
    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            #print('job')
            circ=[]
            for c in self._circuit:
                a=c.bind_parameters(x)
                circ.append(a)
            jobs= self._estimator.run(circuits=circ)
            '''
            job_z= self._estimator.run(circuits=self._circuit[0].bind_parameters(x))
            job_x= self._estimator.run(circuits=self._circuit[1].bind_parameters(x))
            job_y= self._estimator.run(circuits=self._circuit[2].bind_parameters(x))
            '''
            H=0
            #print(jobs.result().quasi_dists)
            '''
            counts_z= util.counts_in_binary_with_padding(job_z.result().quasi_dists[0],self._circuit[0].num_qubits)
            counts_x= util.counts_in_binary_with_padding(job_x.result().quasi_dists[0],self._circuit[1].num_qubits)
            counts_y= util.counts_in_binary_with_padding(job_y.result().quasi_dists[0],self._circuit[2].num_qubits)
            '''
            counts_z= util.counts_in_binary_with_padding(jobs.result().quasi_dists[0],self._circuit[0].num_qubits)
            counts_x= util.counts_in_binary_with_padding(jobs.result().quasi_dists[1],self._circuit[1].num_qubits)
            counts_y= util.counts_in_binary_with_padding(jobs.result().quasi_dists[2],self._circuit[2].num_qubits)
            r=[]
            reg=0
            for op in self._obs[0]:
                value=mthree.utils.expval(counts_x,str(op))
                r.append(value)
                reg+=1/len(self._graph.nodes())*(np.tanh(np.abs(self._alpha*value))-self._gamma)**2  
            for op in self._obs[1]:
                value=mthree.utils.expval(counts_y,str(op))
                r.append(value)
                reg+=1/len(self._graph.nodes())*(np.tanh(np.abs(self._alpha*value))-self._gamma)**2  
            for op in self._obs[2]:
                value=mthree.utils.expval(counts_z,str(op))
                r.append(value)
                reg+=1/len(self._graph.nodes())*(np.tanh(np.abs(self._alpha*value))-self._gamma)**2  
            for edge in self._graph.edges():
                H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.tanh(self._alpha*r[edge[0]-1])*np.tanh(self._alpha*r[edge[1]-1]))/2
            reg=self._beta*self._v*reg**2
            #print(H)
            #print('end')
            if self._callback is not None:
                self._callback([-H,reg,[x]])
            H+=reg
            #print(H)
            return H
        
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        job_z= self._estimator.run(circuits=self._circuit[0].bind_parameters(res.x))
        job_x= self._estimator.run(circuits=self._circuit[1].bind_parameters(res.x))
        job_y= self._estimator.run(circuits=self._circuit[2].bind_parameters(res.x))
        H=0
        counts_z= util.counts_in_binary_with_padding(job_z.result().quasi_dists[0],self._circuit[0].num_qubits)
        counts_x= util.counts_in_binary_with_padding(job_x.result().quasi_dists[0],self._circuit[0].num_qubits)
        counts_y= util.counts_in_binary_with_padding(job_y.result().quasi_dists[0],self._circuit[0].num_qubits)
        i=0
        r=[]
        for op in self._obs[0]:
            value=mthree.utils.expval(counts_x,str(op))
            r.append(value)
        for op in self._obs[1]:
            value=mthree.utils.expval(counts_y,str(op))
            r.append(value)
        for op in self._obs[2]:
            value=mthree.utils.expval(counts_z,str(op))
            r.append(value)

        for edge in self._graph.edges():
            H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(r[edge[0]-1])*np.sign(r[edge[1]-1]))/2
        string=[]
        for i in range(len(self._graph.nodes())):
            if np.sign(self._alpha*r[i])<0:
                string+='0'
            else:
                string+='1'
        print(cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result    


class StatevectorVQE(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,min,shots=None,initial_parameters=None,callback=None):
        self._estimator=estimator
        self._graph=graph
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots

        
    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            circ=self._circuit.bind_parameters(x)
            job_statevector = self._estimator.run(circ)
            result = job_statevector.result()
            statevector = list(result.get_statevector(circ))
            #print(statevector)
            phases = [(cmath.phase(z)) for z in statevector]
            H=0
            for edge in self._graph.edges():
                H-=self._graph[edge[0]][edge[1]].get('weight')*(1-2**(self._circuit.num_qubits-1)*(np.conjugate(statevector[edge[0]-1])*statevector[edge[1]-1]+np.conjugate(statevector[edge[1]-1])*statevector[edge[0]-1]))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.tanh(np.cos(phases[edge[0]-1]))*np.tanh(np.cos(phases[edge[1]-1])))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(np.cos(phases[edge[0]-1]))*(np.cos(phases[edge[1]-1])))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(phases[edge[0]-1]/np.pi)*(phases[edge[1]-1]/np.pi))/2
            if self._callback is not None:

                self._callback([H,phases])

            return H
            #return -np.exp(-H/min)
            #return -np.exp(-H/min*100)
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            #x0 = np.pi/2 * np.random.rand(self._circuit.num_parameters)
            x0 = np.pi/2 * np.ones(self._circuit.num_parameters)
            
        else:
            x0=self.initial_parameters
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        circ=self._circuit.bind_parameters(res.x)
        job_statevector = self._estimator.run(circ)
        result = job_statevector.result()
        statevector = list(result.get_statevector(circ))
        phases = [(cmath.phase(z)) for z in statevector]
        print(phases)
        H=0
        for i,edge in enumerate(self._graph.edges()):
            H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(np.cos(phases[edge[0]-1]))*np.sign(np.cos(phases[edge[1]-1])))/2
            #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(np.cos(phases[edge[0]-1]-phases[edge[1]-1])))/2
            #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(phases[edge[0]-1]/np.pi)*np.sign(phases[edge[1]-1]/np.pi))/2
        string=np.ones(len(self._graph.nodes))
        
        string=[]
        for phase in phases:
            if np.cos(phase)<0:
                string+='0'
            else:
                string+='1'
        print('rounding')
        print(cost_string(self._graph,string))
        string=np.ones(len(self._graph.nodes))*2
        string[list(self._graph.edges)[0][0]-1]=0
        for vertex in string:
            for edge in self._graph.edges():
                #print(string,edge[0]-1,edge[1]-1,np.sign(np.cos(phases[edge[0]-1]-phases[edge[1]-1])))
                if bool(string[edge[0]-1]!=2) + bool(string[edge[1]-1]!=2)==1:
                    
                    if string[edge[0]-1]==2:
                        if  np.sign(np.cos(phases[edge[0]-1]-phases[edge[1]-1]))>0:
                            string[edge[0]-1]=string[edge[1]-1]
                        else:
                            string[edge[0]-1]=(string[edge[1]-1]+1)%2
                    else:
                        if  np.sign(np.cos(phases[edge[0]-1]-phases[edge[1]-1]))>0:
                            string[edge[1]-1]=string[edge[0]-1]
                        else:
                            string[edge[1]-1]=(string[edge[0]-1]+1)%2
                    
        
        print('energy')
        print(-H)
        print('cost_string')
        print(cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        #result.eigenvalue = (H-max)/(min-max)
        result.eigenvalue = cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result
    

class PhaseEncoding_VQE(MinimumEigensolver):

    
    def __init__(self,graph,estimator, circuit, optimizer,min,shots=None,initial_parameters=None,callback=None):
        self._estimator=estimator
        self._graph=graph
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots

        
    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            circ=self._circuit.bind_parameters(x)
            job_statevector = self._estimator.run(circ)
            result = job_statevector.result()
            statevector = list(result.get_statevector(circ))
            phases = [(cmath.phase(z)) for z in statevector]
            H=0
            for edge in self._graph.edges():
                H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.cos(phases[int(edge[0])-1]-phases[int(edge[1])-1]))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.tanh(np.cos(phases[edge[0]-1]))*np.tanh(np.cos(phases[edge[1]-1])))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(np.cos(phases[edge[0]-1]))*(np.cos(phases[edge[1]-1])))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(phases[edge[0]-1]/np.pi)*(phases[edge[1]-1]/np.pi))/2
            if self._callback is not None:

                self._callback([-H,phases])

            return H
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            #x0 = np.pi/2 * np.random.rand(self._circuit.num_parameters)
            x0 = np.pi/2 * np.ones(self._circuit.num_parameters)
            
        else:
            x0=self.initial_parameters
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        circ=self._circuit.bind_parameters(res.x)
        job_statevector = self._estimator.run(circ)
        result = job_statevector.result()
        statevector = list(result.get_statevector(circ))
        phases = [(cmath.phase(z)) for z in statevector]
        print(phases)
        H=0
        for i,edge in enumerate(self._graph.edges()):
            H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(np.cos(phases[int(edge[0])-1]))*np.sign(np.cos(phases[int(edge[1])-1])))/2
            #H-=self._graph[edge[0]][edge[1]].get('weight')*np.sign(np.cos(phases[edge[0]-1]-phases[edge[1]-1]))
            #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(phases[edge[0]-1]/np.pi)*np.sign(phases[edge[1]-1]/np.pi))/2
        
        string=[]
        string=[]
        for phase in phases:
            if np.sign(np.cos(phase))<0:
                string+='0'
            else:
                string+='1'
        print('energy')
        print(-H)
        print('cost_string')
        print(cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        #result.eigenvalue = (H-max)/(min-max)
        result.eigenvalue = cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result
    


class Densitymatrix_VQE(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,min,shots=None,initial_parameters=None,callback=None):
        self._estimator=estimator
        self._graph=graph
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        #self._num_dens=num_dens

        
    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            circ=self._circuit.bind_parameters(x)
            job_statevector = self._estimator.run(circ)
            result = job_statevector.result()
            #dens = np.sum(np.diag(result.results[0].data.density_matrix.data))*np.array(result.results[0].data.density_matrix.data)
            #print(dens)
            dens = np.array(result.results[0].data.density_matrix.data)
            H=0
            for edge in self._graph.edges():
                H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.cos(cmath.phase(dens[edge[0]-1][edge[1]-1])))/2
            if self._callback is not None:
                #print(H)
                self._callback(-H)

            return H
            #return -np.exp(-H/min)
            #return -np.exp(-H/min*100)
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            #x0 = np.pi/2 * np.random.rand(self._circuit.num_parameters)
            x0 = np.pi/2 * np.ones(self._circuit.num_parameters)
            
        else:
            x0=self.initial_parameters
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        circ=self._circuit.bind_parameters(res.x)
        job_statevector = self._estimator.run(circ)
        result = job_statevector.result()
        dens = np.array(result.results[0].data.density_matrix.data)
        #print(dens)
        string=np.ones(len(self._graph.nodes))*2
        string[list(self._graph.edges)[0][0]-1]=0
        for vertex in string:
            for edge in self._graph.edges():
                #print(string,edge[0]-1,edge[1]-1,np.sign(np.cos(phases[edge[0]-1]-phases[edge[1]-1])))
                if bool(string[edge[0]-1]!=2) + bool(string[edge[1]-1]!=2)==1:
                    
                    if string[edge[0]-1]==2:
                        if  np.sign(np.cos(cmath.phase(dens[edge[0]-1][edge[1]-1])))>0:
                            string[edge[0]-1]=string[edge[1]-1]
                        else:
                            string[edge[0]-1]=(string[edge[1]-1]+1)%2
                    else:
                        if  np.sign(np.cos(cmath.phase(dens[edge[0]-1][edge[1]-1])))>0:
                            string[edge[1]-1]=string[edge[0]-1]
                        else:
                            string[edge[1]-1]=(string[edge[0]-1]+1)%2
        print('cost_string')
        print(cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        #result.eigenvalue = (H-max)/(min-max)
        result.eigenvalue = cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result
    

class QRAO_efficient_VQE(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,min,alpha,beta,v,gamma,shots=None,initial_parameters=None,callback=None):
        self._graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._problem=[]
        self._alpha=alpha
        self._beta=beta 
        self._gamma=gamma
        self._v=v
        for op,coeff in util.operator_vertex(self._graph):
            self._problem.append(op)

    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            #print('job')
            circ=[]
            for c in self._circuit:
                a=c.bind_parameters(x)
                circ.append(a)
            jobs= self._estimator.run(circuits=circ)
            N_qubits=len(self._problem[0])
            H=0
            H_round=0
            H_tanh=0
            reg=0

            counts_z= util.counts_in_binary_with_padding(jobs.result().quasi_dists[0],N_qubits)
            counts_x= util.counts_in_binary_with_padding(jobs.result().quasi_dists[1],N_qubits)
            counts_y= util.counts_in_binary_with_padding(jobs.result().quasi_dists[2],N_qubits)
            
            exps_vertex=np.zeros(len(self._graph.nodes()))
            for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    #print(counts_x)
                    #print(ops)
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    #print(counts_y)
                    #print(ops)
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    #print(counts_z)
                    #print(ops)
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))

            for value in exps_vertex:
                reg+=1/len(self._graph.nodes())*(np.abs(value)-self._gamma)**2  
                #reg+=1/len(self._graph.nodes())*(np.tanh(np.abs(self._alpha*value))-self._gamma)**2  
            #print(exps_vertex)
            for edge in self._graph.edges():
                H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(exps_vertex[int(edge[0])-1])*(exps_vertex[int(edge[1])-1]))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.tanh(self._alpha*exps_vertex[edge[0]-1])*np.tanh(self._alpha*exps_vertex[edge[1]-1]))/2
                H_tanh-=self._graph[edge[0]][edge[1]].get('weight')*(1-(np.tanh(self._alpha*exps_vertex[int(edge[0])-1]))*np.tanh((self._alpha*exps_vertex[int(edge[1])-1])))/2

                H_round+=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[int(edge[0])-1])*np.sign(exps_vertex[int(edge[1])-1]))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(exps_vertex[edge[0]-1])*(exps_vertex[edge[1]-1]))/2
            reg=self._beta*self._v*reg
            #print(H)
            #print('end')
            if self._callback is not None:
                print('approxs')
                print(-H/self._min)
                print(H_round/self._min)
                self._callback([-H,reg,H_round,exps_vertex,H_tanh])
            H+=reg

            #print(H)
            return H
        
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.ones(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        N_qubits=len(self._problem[0])
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        job_z= self._estimator.run(circuits=self._circuit[0].bind_parameters(res.x))
        job_x= self._estimator.run(circuits=self._circuit[1].bind_parameters(res.x))
        job_y= self._estimator.run(circuits=self._circuit[2].bind_parameters(res.x))
        H=0
        #print('here')
        counts_z= util.counts_in_binary_with_padding(job_z.result().quasi_dists[0],N_qubits)
        counts_x= util.counts_in_binary_with_padding(job_x.result().quasi_dists[0],N_qubits)
        counts_y= util.counts_in_binary_with_padding(job_y.result().quasi_dists[0],N_qubits)
        i=0

        exps_vertex=np.zeros(len(self._graph.nodes()))
        for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    #print(counts_x)
                    #print(ops)
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    #print(counts_y)
                    #print(ops)
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    #print(counts_z)
                    #print(ops)
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))


        for edge in self._graph.edges():
            #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[edge[0]-1])*np.sign(exps_vertex[edge[1]-1]))/2
            H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[int(edge[0])-1])*np.sign(exps_vertex[int(edge[1])-1]))/2
        string=[]
        for i in range(len(self._graph.nodes())):
            if np.sign(self._alpha*exps_vertex[i])<0:
                string+='0'
            else:
                string+='1'
        #print(util.cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result
    
import mthree
from qiskit.algorithms import MinimumEigensolver, VQEResult
import cmath
from util import cost_string



class QRAO_nonlinear_VQE(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,min,alpha,beta,v,gamma,shots=None,initial_parameters=None,callback=None):
        self._graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._problem=[]
        self._alpha=alpha
        self._beta=beta 
        self._gamma=gamma
        self._v=v
        for op,coeff in util.operator_vertex(self._graph):
            self._problem.append(op)

    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            #print('job')
            circ=[]
            for c in self._circuit:
                a=c.bind_parameters(x)
                circ.append(a)
            jobs=self._estimator.run(circuits=circ)
            N_qubits=len(self._problem[0])
            H=0
            H_tanh=0
            H_round=0
            reg=0

            
            counts_z= util.counts_in_binary_with_padding(jobs.result().quasi_dists[0],N_qubits)
            counts_x= util.counts_in_binary_with_padding(jobs.result().quasi_dists[1],N_qubits)
            counts_y= util.counts_in_binary_with_padding(jobs.result().quasi_dists[2],N_qubits)
            
            exps_vertex=np.zeros(len(self._graph.nodes()))
            for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    #print(counts_x)
                    #print(ops)
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    #print(counts_y)
                    #print(ops)
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    #print(counts_z)
                    #print(ops)
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))
            #print(exps_vertex)
            for value in exps_vertex:
                reg+=1/len(self._graph.nodes())*(np.abs(np.tanh(value))-self._gamma)**2  
                #reg+=1/len(self._graph.nodes())*(np.tanh(np.abs(self._alpha*value))-self._gamma)**2  
            #print(exps_vertex)
            for edge in self._graph.edges():
                H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(exps_vertex[int(edge[0])-1])*(exps_vertex[int(edge[1])-1]))/2
                H_tanh-=self._graph[edge[0]][edge[1]].get('weight')*(1-(np.tanh(self._alpha*exps_vertex[int(edge[0])-1]))*np.tanh((self._alpha*exps_vertex[int(edge[1])-1])))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.tanh(self._alpha*exps_vertex[edge[0]-1])*np.tanh(self._alpha*exps_vertex[edge[1]-1]))/2
                H_round+=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[int(edge[0])-1])*np.sign(exps_vertex[int(edge[1])-1]))/2
                #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-(exps_vertex[edge[0]-1])*(exps_vertex[edge[1]-1]))/2
            reg=self._beta*self._v*reg
            #print(H)
            #print('end')
            if self._callback is not None:
                print('approxs')
                print(-H/self._min)
                print(H_tanh/self._min)
                print(H_round/self._min)
                self._callback([-H,reg,H_round,exps_vertex,H_tanh])
            H_tanh+=reg

            #print(H)
            return H_tanh
        
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        N_qubits=len(self._problem[0])
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        job_z= self._estimator.run(circuits=self._circuit[0].bind_parameters(res.x))
        job_x= self._estimator.run(circuits=self._circuit[1].bind_parameters(res.x))
        job_y= self._estimator.run(circuits=self._circuit[2].bind_parameters(res.x))
        H=0
        #print('here')
        counts_z= util.counts_in_binary_with_padding(job_z.result().quasi_dists[0],N_qubits)
        counts_x= util.counts_in_binary_with_padding(job_x.result().quasi_dists[0],N_qubits)
        counts_y= util.counts_in_binary_with_padding(job_y.result().quasi_dists[0],N_qubits)
        i=0

        exps_vertex=np.zeros(len(self._graph.nodes()))
        for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    #print(counts_x)
                    #print(ops)
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    #print(counts_y)
                    #print(ops)
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    #print(counts_z)
                    #print(ops)
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))


        for edge in self._graph.edges():
            #H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[edge[0]-1])*np.sign(exps_vertex[edge[1]-1]))/2
            H-=self._graph[edge[0]][edge[1]].get('weight')*(1-np.sign(exps_vertex[int(edge[0])-1])*np.sign(exps_vertex[int(edge[1])-1]))/2
        string=[]
        for i in range(len(self._graph.nodes())):
            if np.sign(self._alpha*exps_vertex[i])<0:
                string+='0'
            else:
                string+='1'
        #print(util.cost_string(self._graph,string))
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = string
        result.eigenvalue = util.cost_string(self._graph,string)
        result.optimal_parameters = res.x
        return result
    

import mthree
from qiskit.algorithms import MinimumEigensolver, VQEResult
import cmath
from util import cost_string

class QRAO_initial_state_VQE(MinimumEigensolver):
    
    def __init__(self,graph,estimator, circuit, optimizer,min,alpha,beta,v,gamma,shots=None,initial_parameters=None,callback=None):
        self._graph=graph
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.initial_parameters=initial_parameters
        self._min=min
        self._shots=shots
        self._problem=[]
        self._alpha=alpha
        self._beta=beta 
        self._gamma=gamma
        self._v=v
        for op,coeff in util.operator_vertex(self._graph):
            self._problem.append(op)

    def compute_minimum_eigenvalue(self,min):
                
        # Define objective function to classically minimize over
        def objective(x):
            #print('job')
            circ=[]
            for c in self._circuit:
                a=c.bind_parameters(x)
                circ.append(a)
            jobs=self._estimator.run(circuits=circ)
            N_qubits=len(self._problem[0])
            H_tanh=0
            reg=0

            
            counts_z= util.counts_in_binary_with_padding(jobs.result().quasi_dists[0],N_qubits)
            counts_x= util.counts_in_binary_with_padding(jobs.result().quasi_dists[1],N_qubits)
            counts_y= util.counts_in_binary_with_padding(jobs.result().quasi_dists[2],N_qubits)
            
            exps_vertex=np.zeros(len(self._graph.nodes()))
            for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    #print(counts_x)
                    #print(ops)
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    #print(counts_y)
                    #print(ops)
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    #print(counts_z)
                    #print(ops)
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))
            for value in exps_vertex:
                reg+=1/len(self._graph.nodes())*(np.abs(np.tanh(value))-self._gamma)**2  
                #reg+=1/len(self._graph.nodes())*(np.tanh(np.abs(self._alpha*value))-self._gamma)**2  
            #print(exps_vertex)
            reg=self._beta*self._v*reg**2
            #print(H)
            #print('end')
            if self._callback is not None:
                self._callback([reg,exps_vertex])
            H_tanh+=reg
            print(H_tanh)
            #print(H)
            return H_tanh
        
        # Select an initial point for the ansatzs' parameters
        if self.initial_parameters is None:
            x0 = np.pi/2 * np.random.rand(self._circuit[0].num_parameters)
            
        else:
            x0=self.initial_parameters
        N_qubits=len(self._problem[0])
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        job_z= self._estimator.run(circuits=self._circuit[0].bind_parameters(res.x))
        job_x= self._estimator.run(circuits=self._circuit[1].bind_parameters(res.x))
        job_y= self._estimator.run(circuits=self._circuit[2].bind_parameters(res.x))
        H=0
        counts_z= util.counts_in_binary_with_padding(job_z.result().quasi_dists[0],N_qubits)
        counts_x= util.counts_in_binary_with_padding(job_x.result().quasi_dists[0],N_qubits)
        counts_y= util.counts_in_binary_with_padding(job_y.result().quasi_dists[0],N_qubits)
        i=0

        exps_vertex=np.zeros(len(self._graph.nodes()))
        for i,ops in enumerate(self._problem):
                if 'X' in ops:
                    #print(counts_x)
                    #print(ops)
                    op=ops.replace('X', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_x,str(op))
                if 'Y' in ops:
                    #print(counts_y)
                    #print(ops)
                    op=ops.replace('Y', 'Z')
                    exps_vertex[i]=mthree.utils.expval(counts_y,str(op))
                if 'Z' in ops:
                    #print(counts_z)
                    #print(ops)
                    exps_vertex[i]=mthree.utils.expval(counts_z,str(ops))
        reg=0
        for value in exps_vertex:
                reg+=1/len(self._graph.nodes())*(np.abs(np.tanh(value))-self._gamma)**2  
                #reg+=1/len(self._graph.nodes())*(np.tanh(np.abs(self._alpha*value))-self._gamma)**2  
            #print(exps_vertex)
        reg=self._beta*self._v*reg**2
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenstate = []
        result.eigenvalue = reg
        result.optimal_parameters = res.x
        return result

def iterative_phase_VQE(max_depth,num_qubits,graph,estimator,optimizer,min,shots=None,initial_parameters=None,callback=None):
    print('gw')
    print(min)
    ansatz=ansatz_circ.ansatz_diag2(num_qubits,1)
    num_parameters=ansatz.num_parameters

    energies=[]
    intermediate_info= []
    def callback(data):
                    intermediate_info.append(data)
    
    custom_vqe = PhaseEncoding_VQE(graph,estimator,ansatz,optimizer,num_qubits,min,initial_parameters=None,callback=callback)
    result= custom_vqe.compute_minimum_eigenvalue(min)
    energies.append(result.eigenvalue)
    for depth in range (2,max_depth+1):
        ansatz=ansatz_circ.ansatz_diag2(num_qubits,depth)
        initial_parameters=list(result.optimal_parameters)+list(np.zeros(num_parameters)*np.pi/20)
        custom_vqe = PhaseEncoding_VQE(graph,estimator,ansatz,optimizer,num_qubits,min,initial_parameters=initial_parameters,callback=callback)
        result= custom_vqe.compute_minimum_eigenvalue(min)
        energies.append(result.eigenvalue)
    
    return intermediate_info,energies



def iterative_state_VQE(max_depth,num_qubits,graph,estimator,optimizer,min,shots=None,initial_parameters=None,callback=None):
    print('gw')
    print(min)
    ansatz=ansatz_circ.ansatz_diag2(num_qubits,1)
    num_parameters=ansatz.num_parameters

    energies=[]
    intermediate_info= []
    def callback(data):
                    intermediate_info.append(data)
    
    custom_vqe = StatevectorVQE(graph,estimator,ansatz,optimizer,num_qubits,min,initial_parameters=None,callback=callback)
    result= custom_vqe.compute_minimum_eigenvalue(min)
    energies.append(result.eigenvalue)
    for depth in range (2,max_depth+1):
        ansatz=ansatz_circ.ansatz_diag2(num_qubits,depth)
        initial_parameters=list(result.optimal_parameters)+list(np.random.rand(num_parameters)*np.pi/20)
        custom_vqe = StatevectorVQE(graph,estimator,ansatz,optimizer,num_qubits,min,initial_parameters=initial_parameters,callback=callback)
        result= custom_vqe.compute_minimum_eigenvalue(min)
        energies.append(result.eigenvalue)
    
    return intermediate_info,energies


def iterative_pauli_VQE(max_depth,graph,estimator,optimizer,n_vertex,num_qubits,num_ancillas,min,alpha,beta,v,gamma,initial_parameters=None):
    print('gw_cut')
    print(min)
    results=[]
    ans=ansatz_circ.ansatz_xy(num_qubits+num_ancillas,1)
    n_param=ans.num_parameters
    obs=util.vertex_to_pauli(n_vertex,num_qubits,num_qubits,num_ancillas)
    intermediate_info= []
    def callback(data):
                    intermediate_info.append(data)

    custom_vqe = PauliEncoding_VQE(estimator,graph,ans,optimizer,obs,num_qubits,min,alpha,beta,v,gamma,initial_parameters=None,callback=callback)
    
    result= custom_vqe.compute_minimum_eigenvalue(min)
    print(result.eigenvalue)
    results.append(result.eigenvalue)
    for i in range (2,max_depth+1):
        ansatz=ansatz_circ.ansatz_xy(num_qubits+num_ancillas,i)
        initial_parameters=list(result.optimal_parameters)
        initial_parameters+=list(np.random.rand(n_param)*np.pi/20)
        custom_vqe = PauliEncoding_VQE(estimator,graph,ansatz,optimizer,obs,num_qubits,min,alpha,beta,v,gamma,initial_parameters=initial_parameters,callback=callback)
        result= custom_vqe.compute_minimum_eigenvalue(min)
        print(result.eigenvalue)
        results.append(result.eigenvalue)
    return intermediate_info,results

def iterative_efficient_pauli_VQE(max_depth,graph,estimator,optimizer,n_vertex,num_qubits,num_ancillas,min,alpha,beta,v,gamma,initial_parameters=None,shots=None):
    print('gw_cut')
    print(min)
    results=[]
    ans=ansatz_circ.ansatz_efficient(num_qubits+num_ancillas,1)
    circuits=ansatz_circ.multibasis_ansatz(ans)
    n_param=ans.num_parameters
    obs=util.vertex_to_pauli_dict(n_vertex,num_qubits,num_qubits,num_ancillas)
    intermediate_info= []
    def callback(data):
                    intermediate_info.append(data)

    custom_vqe = Pauli_efficient_VQE(graph,estimator, circuits, optimizer,obs,num_qubits,min,alpha,beta,v,gamma,shots=None,initial_parameters=None,callback=callback)
    
    result= custom_vqe.compute_minimum_eigenvalue(min)
    #print(result.eigenvalue)
    results.append(result.eigenvalue)
    for i in range (2,max_depth+1):
        ansatz=ansatz_circ.ansatz_efficient(num_qubits+num_ancillas,i)
        circuits=ansatz_circ.multibasis_ansatz(ansatz)
        initial_parameters=list(result.optimal_parameters)
        initial_parameters+=list(np.zeros(n_param)*np.pi/20)
        custom_vqe = Pauli_efficient_VQE(graph,estimator, circuits, optimizer,obs,num_qubits,min,alpha,beta,v,gamma,shots=None,initial_parameters=initial_parameters,callback=callback)
        result= custom_vqe.compute_minimum_eigenvalue(min)
        #print(result.eigenvalue)
        results.append(result.eigenvalue)
    return intermediate_info,results


def iterative_density_VQE(max_depth,num_qubits,num_dens,graph,estimator,optimizer,min,shots=None,initial_parameters=None,callback=None):
    print('gw')
    print(min)
    ansatz=ansatz_circ.ansatz_density(num_qubits,num_dens,1)
    num_parameters=ansatz.num_parameters

    energies=[]
    intermediate_info= []
    def callback(data):
                    intermediate_info.append(data)
    
    custom_vqe = Densitymatrix_VQE(graph,estimator,ansatz, optimizer,min,shots=None,initial_parameters=None,callback=None)
    result= custom_vqe.compute_minimum_eigenvalue(min)
    energies.append(result.eigenvalue)
    for depth in range (2,max_depth+1):
        ansatz=ansatz_circ.ansatz_density(num_qubits,num_dens,depth)
        initial_parameters=list(result.optimal_parameters)+list(np.random.rand(num_parameters)*np.pi/20)
        custom_vqe = Densitymatrix_VQE(graph,estimator, ansatz, optimizer,min,shots=None,initial_parameters=None,callback=None)
        result= custom_vqe.compute_minimum_eigenvalue(min)
        energies.append(result.eigenvalue)
    
    return intermediate_info,energies