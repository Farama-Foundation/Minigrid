from collections import defaultdict

# ruleset = defaultdict(dict)
#
# def get_ruleset():
#     return ruleset
#
# def set_ruleset(_ruleset):
#     global ruleset
#     ruleset = _ruleset


def extract_rule(block_list):
    """
    Take a list of 3 blocks and return the rule object and property if these blocks form a valid rule, otherwise
    return None. Valid rule: 'object', 'is', 'property'
    """
    assert len(block_list) == 3
    for e in block_list:
        if e is None:
            return None

    is_valid = \
        block_list[0].type == 'rule_object' and \
        block_list[1].type == 'rule_is' and \
        block_list[2].type == 'rule_property'

    if is_valid:
        return {
            'object': block_list[0].object,
            'property': block_list[2].property
        }
    else:
        return None


def add_rule(block_list, ruleset):
    """
    If the blocks form a valid rule, add it to the ruleset
    Args:
        block_list: list of 3 blocks
        ruleset: dict with the active rules
    """
    rule = extract_rule(block_list)
    if rule is not None:
        ruleset[rule['property']][rule['object']] = True


def inside_grid(grid, pos):
    """
    Return true if pos is outside the boundaries of the grid
    """
    i, j = pos
    inside_grid = (i >= 0 and i < grid.width) and (j >= 0 and j < grid.height)
    return inside_grid


def extract_ruleset(grid):
    """
    Construct the ruleset from the grid. Called every time a RuleBlock is pushed.
    """
    ruleset = defaultdict(dict)
    # loop through all 'is' blocks
    # for k, e in enumerate(grid.grid):
    for k, e in enumerate(grid):
        if e is not None and e.type == 'rule_is':
            i, j = k % grid.width, k // grid.width
            assert k == j * grid.width + i

            # check for horizontal
            if inside_grid(grid, (i-1, j)) and inside_grid(grid, (i+1, j)):
                left_cell = grid.get(i-1, j)
                right_cell = grid.get(i+1, j)
                add_rule([left_cell, e, right_cell], ruleset)

            # check for vertical rules
            if inside_grid(grid, (i, j-1)) and inside_grid(grid, (i, j+1)):
                up_cell = grid.get(i, j-1)
                down_cell = grid.get(i, j+1)
                add_rule([up_cell, e, down_cell], ruleset)

    return ruleset
