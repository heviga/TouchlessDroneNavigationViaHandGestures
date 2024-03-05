import numpy as np
from scipy.optimize import linprog

#max zisk 30p + 50d
#min - 30p - 50d
#cas: 3p + 4d = 120
#suroviny: 2p + 4d <= 100

c = [-30, -50]

#lava strana obmedzeni
A_eq = [[3, 4]]
A_ub = [[2, 4]]


#prava strana
b_eq = [120]
b_ub = [100]

#nezaporne mnozstva
p_bounds = (0, None)
d_bounds = (0, None)

result_1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[p_bounds, d_bounds])
print("Prvé riešenie:")
print(f"Zisk: {-result_1.fun} EUR")
print(f"Počet kusov prémiovej verzie: {int(result_1.x[0])}")
print(f"Počet kusov deluxe verzie: {int(result_1.x[1])}")

result_2 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[p_bounds, d_bounds], x0=result_1.x, method='highs')
print("\nDruhé riešenie:")
print(f"Zisk: {-result_2.fun} EUR")
print(f"Počet kusov prémiovej verzie: {int(result_2.x[0])}")
print(f"Počet kusov deluxe verzie: {int(result_2.x[1])}")

result_3 = linprog(c, A_ub=A_ub,  b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[p_bounds, d_bounds], x0=result_2.x,  method='highs')
print("\nTretie riešenie:")
print(f"Zisk: {-result_3.fun} EUR")
print(f"Počet kusov prémiovej verzie: {int(result_3.x[0])}")
print(f"Počet kusov deluxe verzie: {int(result_3.x[1])}")
