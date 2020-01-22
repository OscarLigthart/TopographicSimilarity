def compute_jaccard_similarity_score(x, y):
    """
    Jaccard Similarity J (A,B) = | Intersection (A,B) | /
                                    | Union (A,B) |
    """
    intersection_cardinality = len(set(x).intersection(set(y)))
    union_cardinality = len(set(x).union(set(y)))
    return intersection_cardinality / float(union_cardinality)


a = [0, 4, 6, 4, 3]

b = [3, 6, 4, 0, 4]

c = compute_jaccard_similarity_score(a, b)

print(c)