import pytest
import numpy as np
import cupy as cp

def test_basic_operations():
    x_cpu = np.array([1, 2, 3, 4, 5])
    x_gpu = cp.array([1, 2, 3, 4, 5])
    assert cp.allclose(x_gpu, cp.asarray(x_cpu))
    
    assert cp.allclose(x_gpu + 1, cp.asarray(x_cpu + 1))
    assert cp.allclose(x_gpu * 2, cp.asarray(x_cpu * 2))
    
    assert abs(cp.mean(x_gpu).get() - np.mean(x_cpu)) < 1e-6
    assert abs(cp.sum(x_gpu).get() - np.sum(x_cpu)) < 1e-6

def test_matrix_operations():
    A = cp.random.rand(10, 10)
    B = cp.random.rand(10, 10)
    
    C_gpu = cp.matmul(A, B)
    C_cpu = np.matmul(cp.asnumpy(A), cp.asnumpy(B))
    assert cp.allclose(C_gpu, cp.asarray(C_cpu))
    
    assert cp.allclose(A.T, cp.asarray(cp.asnumpy(A).T))

def test_device_memory():
    x = cp.zeros((1000, 1000))
    del x
    cp.get_default_memory_pool().free_all_blocks()
    
    x_cpu = np.random.rand(100, 100)
    x_gpu = cp.asarray(x_cpu)
    x_back = cp.asnumpy(x_gpu)
    assert np.allclose(x_cpu, x_back)

if __name__ == "__main__":
    pytest.main([__file__])
