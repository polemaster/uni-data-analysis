from collections import defaultdict
from decimal import Decimal, getcontext

getcontext().prec = 20  # High precision

def load_graph(path):
    graph = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(":")
            node = parts[0].strip()
            links = parts[1].strip().split() if len(parts) > 1 and parts[1].strip() else []
            graph[node] = links
    return graph

def initialize_ranks(graph, N):
    initial = Decimal(1) / Decimal(N)
    return {node: initial for node in graph}

def pagerank_with_taxation(graph, N, K, B):
    ranks = initialize_ranks(graph, N)
    B = Decimal(str(B))
    N = Decimal(str(N))

    for _ in range(K):
        new_ranks = defaultdict(Decimal)
        for node in graph:
            links = graph[node]
            if links:
                share = B * ranks[node] / Decimal(len(links))
                for dest in links:
                    new_ranks[dest] += share
        for node in graph:
            new_ranks[node] += (Decimal(1) - B) / N
        ranks = new_ranks
    return ranks

def run(graph_path, N, K, B, variant):
    graph = load_graph(graph_path)
    N_decimal = Decimal(str(N))

    if variant == 'taxation':
        ranks = pagerank_with_taxation(graph, N_decimal, K, B)
    else:
        raise ValueError("Only 'taxation' variant supports Decimal high-precision fix")

    for node in sorted(graph):
        rank = f"{ranks.get(node, Decimal(0)):.9f}"
        links = " ".join(graph[node])
        print(f"{node} {rank} {links}".strip())

# Example Usage:
# Replace with actual file paths or adapt to Hadoop environment
# run("deadEnds.txt", N=4, K=3, B=0.8, variant="taxation")

if __name__ == "__main__":
    print("Sample output")
    run("deadEnds.txt", N=4, K=3, B=1, variant="taxation")
    print()
    run("spiderTraps.txt", N=4, K=3, B=1, variant="taxation")
    print()
    print("quiz - q1 (B?):")
    run("links.txt", N=4, K=10, B=0.8, variant="taxation")
    print()
    print("quiz - q2: (A?)")
    run("links.txt", N=4, K=10, B=1, variant="taxation")
    print()
    print("quiz - q3: (D?)")
    run("links.txt", N=4, K=10, B=1, variant="taxation")
