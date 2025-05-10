# -*- coding: utf-8 -*-
def generate_datastructures(inst):
    due_date = {
        (ele['item'], key): ele['due_date'] for key, ele in inst.jobs.items()
    }
    release_date = {
        key: ele['release_date'] for key, ele in inst.jobs.items()
    }

    final_operations = due_date.keys()
    n_positions = len(inst.lst_operations)
    positions = range(n_positions)

    duration_op = {}
    for l in inst.lst_machines:
        duration_op[l] = {}
        for op in inst.lst_operations:
            tmp = inst.df_operations[(inst.df_operations.machines==l)&(inst.df_operations.op==op)]
            if len(tmp) == 0:
                duration_op[l][op] = 0
            else:
                duration_op[l][op] = tmp.iloc[0].duration_h
    
    def machine_setup(m, j0, j1):
        ris = inst.df_setup[(inst.df_setup['machine']== m) & (inst.df_setup['op1']== j0) & (inst.df_setup['op2']== j1)]
        if len(ris) == 0:
            return 0
        else:
            return ris.iloc[0]['time_h']
     
    return due_date, release_date, final_operations, n_positions, positions, duration_op, machine_setup

def compute_starting_interval(df_machine_production, precedence_graph, order, idx_order):

    earliest_starting = {}
    lates_starting = {}
    
    # take the last operarion
    op_to_process = [
        (order['item'], 'na')
    ]
    successors_early_start = order['due_date']
    successors_tardy_start = order['due_date']
    # iterate in the predecessors
    while len(op_to_process) > 0:
        considered_op, parent_op = op_to_process.pop()
        speed_vector = df_machine_production[df_machine_production.op == considered_op].speed.values
        max_speed = max(speed_vector)
        min_speed = min(speed_vector)
        # if there a parent node
        if parent_op != 'na':
            successors_early_start = earliest_starting[(parent_op, idx_order)]
            successors_tardy_start = lates_starting[(parent_op, idx_order)]
        # compute the 
        earliest_starting[(considered_op, idx_order)] = successors_early_start - order['qnt'] / min_speed
        lates_starting[(considered_op, idx_order)] = successors_tardy_start - order['qnt'] /  max_speed
        # Add to op_to_process the parents operations
        for ele in precedence_graph.predecessors(considered_op):
            op_to_process.append(
                (ele, considered_op)
            )
    return earliest_starting, lates_starting


def get_subtree_product(g, start_node):
    stack = [start_node]
    ws = []
    while stack:
        node = stack.pop()
        ws.append(node)
        preds = g.predecessors(node)
        stack += preds
    return ws

def complete_data(
    index, order, tree_order, df_machine_production,
    data_operations,
    earliest_starting, lates_starting,
    raw_op_forest, operations_forest
    ):

    for op in tree_order: # tree order è [ultimo -> penultimo -> ...]
        df_tmp = df_machine_production[df_machine_production['op']==op]
        for _, row_machine in df_tmp.iterrows():
            # i = 0
            # row_machine = df_tmp.iloc[0]
            data_operations['op'].append(
                (op, index)
            )
            data_operations['order'].append(
                index
            )
            data_operations['id_order'].append(
                order['id_order']
            )

            data_operations['speed'].append(row_machine['speed'])
            data_operations['machines'].append(row_machine['machine'])
            duration_h = order['qnt'] / row_machine['speed']
            data_operations['duration_h'].append(duration_h)
            data_operations['importance'].append(1)
            data_operations['release_date'].append(0)
            data_operations['job_due_date'].append(
                order['due_date']
            )
            data_operations['earliest_starting'].append(
                earliest_starting[(op,index)]
            )
            data_operations['lates_starting'].append(
                lates_starting[(op,index)]
            )
            # tree_order[2] -> tree_order[1] -> tree_order[0]
            data_operations['op_to_last'].append(
                tree_order.index(op)
            )

    # CREATION GRAPH operations_forest
    tmp_subgraph = raw_op_forest.subgraph(tree_order)
    for node in tmp_subgraph.nodes():
        predecessors = raw_op_forest.predecessors(node)
        for pred in predecessors:
            data_edge = raw_op_forest.get_edge_data(pred, node)
            operations_forest.add_edge(
                (pred, index),
                (node, index),
                max_waiting=data_edge['max_waiting'],
                min_waiting=data_edge['min_waiting']
            )
            operations_forest.nodes()[(pred, index)]['color'] = 'blue'
            operations_forest.nodes()[(node, index)]['color'] = 'blue'

        if node == order['item']:
            if (node, index) not in operations_forest.nodes():
                operations_forest.add_node(
                    (node, index)
                )
            operations_forest.nodes()[
                (node, index)
            ]['color'] = 'red'
