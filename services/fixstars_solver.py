from amplify import VariableGenerator, solve
from amplify.client import FixstarsClient
import logging


def solve_with_fixstars(Q, api_key):

    # 変数数
    max_index = max(max(i, j) for i, j in Q.keys())
    n = max_index + 1

    gen = VariableGenerator()
    x = gen.array("Binary", n)

    objective = 0
    for (i, j), val in Q.items():
        if i == j:
            objective += val * x[i]
        else:
            objective += val * x[i] * x[j]

    client = FixstarsClient()
    client.token = api_key

    result = solve(objective, client)

    solution = result[0]

    values = solution.values
    energy = solution.objective  # ← ここが修正点

    selected_indices = [i for i in range(n) if values[x[i]] == 1]

    return {"selected_indices": selected_indices, "energy": energy}
