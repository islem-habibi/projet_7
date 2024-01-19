from sklearn.metrics import confusion_matrix
cost_fn=10
cost_fp= 1
def score_metier(y_test, y_pred, cost_fn, cost_fp):

  """
  l'objectif est de minimiser les coûts associés aux faux négatifs (FN) et faux positifs (FP),
  particulièrement en cas de déséquilibre entre les deux cibles.

  """

  cm = confusion_matrix(y_test, y_pred)
  fn = cm[1, 0]  # Faux Négatifs
  fp = cm[0, 1]  # Faux Positifs

  # Calculer le score métier
  score_metier = cost_fn * fn + cost_fp * fp  #plus le score metier est faible , plus le model est meilleur

  return score_metier