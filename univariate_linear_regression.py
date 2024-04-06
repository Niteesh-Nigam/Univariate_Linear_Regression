import numpy as np
import matplotlib.pyplot as plt
import math


# dataset = pd.read.csv('./test.csv')
data = np.genfromtxt('./test.csv', delimiter=',', skip_header=1)
# print(data)
x_train= np.array(data[:,0])
y_train = np.array(data[:,1])

def compute_model_output(x, w, b):
    m=x.shape[0]
    f_wb= np.zeros(m)
    for i in range(m):
        f_wb[i] = w* x[i] + b
    return f_wb

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) **2
        cost_sum += cost
    total_cost = (1 / (2*m)) *cost_sum
    return  total_cost

def compute_gradient(x, y, w, b):
    m=x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range (m):
        f_wb = w*x[i] + b
        dj_dw_i = (f_wb -y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db = dj_db_i
        dj_dw = dj_dw/m
        dj_db = dj_db/m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []
    w=w_in
    b=b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        b_tmp= b- alpha * dj_db
        w_tmp= w- alpha * dj_dw
        b= b_tmp
        w= w_tmp

                # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
        

    return w, b, J_history, p_history


w_init = 0
b_init = 0
iterations =100
tmp_alpha = 0.005
w_final, b_final, J_history, p_history = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"final vlaues of parameters after gradient descent: w: {w_final} b: {b_final}")
y_pred=np.zeros_like(y_train)
for i in range(y_pred.shape[0]):
    y_pred[i] = w_final*x_train[i]+ b_final
# y_plt1 = [w_final*x_train[0]+ b_final, w_final*x_train[-1] + b_final]
# x_plt1 = [x_train[0], x_train[-1]]

plt.scatter(x_train, y_train, marker='x' ,c='b')
plt.plot(x_train, y_pred, c= 'r' )

plt.title("Univariate Linear Regression")
plt.ylabel("target y values")
plt.xlabel("input x values")
plt.show()



plt.plot(J_history[:100])
# ax2.plot(1000 + np.arange(len(J_history[1000:])), J_history[1000:])
plt.title("Cost vs. iteration(start)")
plt.ylabel('Cost')  
plt.xlabel('iteration step')
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
x=np.zeros(len(p_history))
y=np.zeros(len(p_history))
v=np.zeros(len(p_history))
for i in range(len(p_history)):
        x[i] = p_history[i][0]
        y[i] = p_history[i][1]
        v[i] = J_history[i]
ax1.plot(x,v)
ax2.plot(y,v)
ax1.set_title("Cost vs. W");  ax2.set_title("Cost vs. b")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('parameter W')  ;  ax2.set_xlabel('parameter b') 
plt.show()
