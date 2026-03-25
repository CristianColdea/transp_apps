import numpy as np

def find_zero_permutations(matrix):
    """
    Finds all sequences of zeros such that there is exactly one 
    zero in each row and each column.
    """
    n = matrix.shape[0]
    
    # Pre-compute the column indices of zeros for each row to speed up lookup
    zero_positions = [np.where(matrix[r] == 0)[0] for r in range(n)]
    
    all_sequences = []

    def backtrack(row, used_cols, current_seq):
        if row == n:
            all_sequences.append(list(current_seq))
            return

        for col in zero_positions[row]:
            if col not in used_cols:
                # Explore this branch
                used_cols.add(col)
                current_seq.append((row, col))
                
                backtrack(row + 1, used_cols, current_seq)
                
                # Backtrack
                current_seq.pop()
                used_cols.remove(col)

    backtrack(0, set(), [])
    return all_sequences

# --- Example Usage ---
# 0 = Zero, 1 = Non-zero (or null)
mat = np.array([
    [3, 3, 1, 0],
    [6, 3, 0, 4],
    [0, 5, 2, 0],
    [3, 0, 3, 6]
])

sequences = find_zero_permutations(mat)

print(f"sequences: {sequences}")

print(f"Found {len(sequences)} valid sequence(s):")
for seq in sequences:
    print(seq)
