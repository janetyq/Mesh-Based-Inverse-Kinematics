# ALL THE METHODS OF SOLVING
def solve_for_target_feature(f_target):
    '''
    Finds best vertex positions to match a given feature vector
    '''
    x = np.linalg.pinv(G_tilda) @ (f_target - C)
    x = constraint_post_processing(x)
    return x

def solve_for_constraints(k=0):
    '''
    Finds best blend of example meshes that satisfies the constrained vertices
    '''
    mean_f = np.mean(M, axis=1, keepdims=True)
    Sx = np.block([np.eye(3*n_tilda), np.zeros((3*n_tilda, N))])
    Sw = np.block([np.zeros((N, 3*n_tilda)), np.eye(N)])
    A = G_tilda @ Sx - (M @ Sw)
    b = -C + mean_f
    if k != 0:
        Gamma = k * Sw
        xw = np.linalg.inv(A.T @ A + Gamma.T @ Gamma) @ A.T @ b
        error = np.linalg.norm(A @ xw - b)**2 + np.linalg.norm(Gamma @ xw)**2
    else:
        xw = np.linalg.pinv(A) @ b
        error = np.linalg.norm(A @ xw - b)**2
    x_result = xw[:3*n_tilda]
    w_result = xw[3*n_tilda:]
     
    x_result = constraint_post_processing(x_result)
    return x_result, w_result, error