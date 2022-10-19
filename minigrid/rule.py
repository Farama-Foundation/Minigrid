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
    """
    rule = extract_rule(block_list)
    if rule is not None:
        ruleset[rule['property']][rule['object']] = True


def extract_ruleset(grid):
    """
    Construct the ruleset from the grid. Called every time a RuleBlock is pushed.
    """
    ruleset = defaultdict(dict)
    # loop through all 'is' blocks
    for k, e in enumerate(grid.grid):
        if e is not None and e.type == 'rule_is':
            i, j = k % grid.width, k // grid.width
            assert k == j * grid.width + i

            # get neighboring cells
            left_cell = grid.get(i-1, j)
            right_cell = grid.get(i+1, j)
            up_cell = grid.get(i, j-1)
            down_cell = grid.get(i, j+1)

            # check for horizontal and vertical rules
            add_rule([left_cell, e, right_cell], ruleset)
            add_rule([up_cell, e, down_cell], ruleset)
            # overwrite the global ruleset
    # set_ruleset(ruleset)
    return ruleset
