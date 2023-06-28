import pytest
from k_centers import *

def test1_calc_minkowski_distance():
    x = np.array([1,2,3])
    y = np.array([4,5,6])
    assert calc_minkowski_distance(x,y,1) == 9
    assert calc_minkowski_distance(x,y,2) == np.sqrt(27)
    assert np.isclose(calc_minkowski_distance(x,y,3),np.power(81,1/3))

def test2_calc_minkowski_distance():
    x = np.array([0,2,3,4])
    y = np.array([2,4,3,7])
    assert np.isclose(calc_minkowski_distance(x,y,3),3.50339806)

def test3_calc_minkowski_distance():
    x = np.array([10,20,15,10,5])
    y = np.array([12,24,18,8,7])
    assert calc_minkowski_distance(x,y,1) == 13
    assert np.isclose(calc_minkowski_distance(x,y,2),6.082762530298219)

def test1_calc_distance_matrix():
    X = np.array([[1,2,3],[4,5,6],[7,8,9]])
    D = calc_distance_matrix(X,1)
    assert D.shape == (3,3)
    assert D[0,0] == 0
    assert D[1,1] == 0
    assert D[2,2] == 0
    assert D[0,1] == 9
    assert D[0,2] == 18
    assert D[1,0] == 9
    assert D[1,2] == 9
    assert D[2,0] == 18
    assert D[2,1] == 9

def test2_calc_distance_matrix():
    X = np.array([[1,2,3,4,6],[1,2,3,4,5]])
    D = calc_distance_matrix(X,3)
    assert D.shape == (2,2)

def test_farthest_point_from_centers():
    """
       0 1 2
    0 [0,9,18]
    1 [9,0,9]
    2 [18,9,0]
    """
    D = np.array([[0,9,18],[9,0,9],[18,9,0]])
    centers = [0]
    assert farthest_point_from_centers(D,centers) == 2
    centers = [0,1]
    assert farthest_point_from_centers(D,centers) == 2
    with pytest.raises(Exception) as excinfo:
        centers = [0,1,2]
        assert farthest_point_from_centers(D,centers) == 0
        assert str(excinfo.value) == "farthest_point == -1"