#pytest test_unitaire.py

from score_metier_func import score_metier
from sklearn.metrics import confusion_matrix

cost_fn=10
cost_fp= 1
y_test1=[0, 1, 1, 1, 0, 0]
y_test2=[0, 5, 2, 1, 10, 6, 7, 0]
y_test3=["d", "c", "b", "a", "c", "a", 'c', "c", "b", "d"]

y_pred1=[0, 1, 0, 0, 0, 1]
y_pred2=[0, 1, 0, 10, 7, 1, 8, 3]
y_pred3=["a", "c", "c", "b", "a", "a", "b", "v", "a", "a"]

def test_score_metier():
    assert score_metier(y_test1, y_pred1, cost_fn, cost_fp)==21
    assert score_metier(y_test2, y_pred2, cost_fn, cost_fp)== 0
    assert score_metier(y_test3, y_pred3, cost_fn, cost_fp)== 11
    
