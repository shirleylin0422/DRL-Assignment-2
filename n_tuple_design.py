def get_patterns():
    patterns = []
    # Horizontal rows (4 patterns)
    for i in range(4):
        pattern = [(i, j) for j in range(4)]
        patterns.append(pattern)
    
    # Vertical columns (4 patterns)
    for j in range(4):
        pattern = [(i, j) for i in range(4)]
        patterns.append(pattern)
    
    # 2x2 squares (9 patterns)
    for i in range(3):
        for j in range(3):
            pattern = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
            patterns.append(pattern)
    return patterns