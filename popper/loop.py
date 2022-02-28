#!/usr/bin/env python3

import logging
from operator import neg
from os import stat
import sys
from . util import Settings, Stats, timeout, parse_settings, format_program
from . asp import ClingoGrounder, ClingoSolver
from . tester import Tester
from . constrain import Constrain
from . generate import generate_program
from . core import Grounding, Clause, is_subset, is_generalization_of, is_specialization_of

class Outcome:
    ALL = 'all'
    SOME = 'some'
    NONE = 'none'

class Con:
    GENERALISATION = 'generalisation'
    SPECIALISATION = 'specialisation'
    REDUNDANCY = 'redundancy'
    BANISH = 'banish'

OUTCOME_TO_CONSTRAINTS = {
        (Outcome.ALL, Outcome.NONE)  : (Con.BANISH,),
        (Outcome.ALL, Outcome.SOME)  : (Con.GENERALISATION,),
        (Outcome.SOME, Outcome.NONE) : (Con.SPECIALISATION,),
        (Outcome.SOME, Outcome.SOME) : (Con.SPECIALISATION, Con.GENERALISATION),
        (Outcome.NONE, Outcome.NONE) : (Con.SPECIALISATION, Con.REDUNDANCY),
        (Outcome.NONE, Outcome.SOME) : (Con.SPECIALISATION, Con.REDUNDANCY, Con.GENERALISATION)
    }

def ground_rules(stats, grounder, max_clauses, max_vars, clauses):
    out = set()
    for clause in clauses:
        head, body = clause
        # find bindings for variables in the constraint
        assignments = grounder.find_bindings(clause, max_clauses, max_vars)

        # keep only standard literals
        body = tuple(literal for literal in body if not literal.meta)

        # ground the clause for each variable assignment
        for assignment in assignments:
            out.add(Grounding.ground_clause((head, body), assignment))
    
    stats.register_ground_rules(out)

    return out

def decide_outcome(conf_matrix):
    tp, fn, tn, fp = conf_matrix
    if fn == 0:
        positive_outcome = Outcome.ALL # complete
    elif tp == 0 and fn > 0:
        positive_outcome = Outcome.NONE # totally incomplete
    else:
        positive_outcome = Outcome.SOME # incomplete
    if fp == 0:
        negative_outcome = Outcome.NONE  # consistent
    elif tn == 0 and fp > 0:
        negative_outcome = Outcome.ALL # totally inconsistent
    else:
        negative_outcome = Outcome.SOME # inconsistent

    return (positive_outcome, negative_outcome)

def build_rules(settings, stats, constrainer, tester, program, before, min_clause, outcome):
    (positive_outcome, negative_outcome) = outcome
    rules = set()

    gens_pruned_flag = False # JW: Avoid redundant constraints
    specs_pruned_flag = False 
    # JW: Noisy setting - Sound Constraints, Size Constraints, and Minimal Constraints
    if settings.noisy:
        # JW: Sound Constraints
        (new_tp, new_fn, new_tn, new_fp) = stats.all_programs[-1].conf_matrix
        new_score = calc_score((new_tp, new_fn, new_tn, new_fp))
        num_pos = new_tp + new_fn
        num_neg = new_tn + new_fp

        for old_program in stats.all_programs[:-1]:
            (old_tp, old_fn, old_tn, old_fp) = old_program.conf_matrix

            if new_tp == old_tp and is_generalization_of(program, old_program.program) and len(program) > len(old_program.program):
                if is_subset(old_program.program, program):
                    # JW: Prune GENS and SPECS of NEW program
                    rules.update(constrainer.generalisation_constraint(program, before, min_clause))
                    gens_pruned_flag = True
                    rules.update(constrainer.specialisation_constraint(program, before, min_clause))
                    specs_pruned_flag = True
                    break
                else:
                    # JW: Prune GENS of NEW program
                    rules.update(constrainer.generalisation_constraint(program, before, min_clause))
                    gens_pruned_flag = True

            if not gens_pruned_flag and new_tn == old_tn and is_specialization_of(program, old_program.program):
                # JW: Prune GENS of NEW Program
                rules.update(constrainer.generalisation_constraint(program, before, min_clause))
                gens_pruned_flag = True

        i = 0
        while i < len(stats.can_prune_gens):
            old_program = stats.can_prune_gens[i]
            old_score = calc_score(old_program.conf_matrix)
            if new_score - old_score > num_pos - old_tp:
                # JW: Prune GENS of OLD program
                rules.update(constrainer.generalisation_constraint(old_program.program, before, min_clause))
                del stats.can_prune_gens[i]
                i -= 1
            i += 1

        i = 0
        while i < len(stats.can_prune_specs):
            old_program = stats.can_prune_specs[i]
            old_score = calc_score(old_program.conf_matrix)
            if new_score - old_score > num_neg - old_tn:
                # JW: Prune SPECS of OLD 
                rules.update(constrainer.specialisation_constraint(old_program.program, before, min_clause))
                del stats.can_prune_specs[i]
                i -= 1
            i += 1

        # JW: Minimal Constraints - Only prune generalizations when entailing all negative examples. Only prune specializations when entailing no positive examples
        # JW: Still prune generalisations when entailing all positive examples. Still prune specializations when entailing no negative examples
        if positive_outcome == Outcome.SOME and negative_outcome == Outcome.ALL:
            positive_outcome = Outcome.ALL
        elif positive_outcome == Outcome.NONE and negative_outcome == Outcome.SOME:
            negative_outcome = Outcome.NONE
        elif positive_outcome == Outcome.SOME and negative_outcome == Outcome.SOME:
            positive_outcome = Outcome.ALL
            negative_outcome = Outcome.NONE

    # RM: If you don't use these two lines you need another three entries in the OUTCOME_TO_CONSTRAINTS table (one for every positive outcome combined with negative outcome ALL).
    if negative_outcome == Outcome.ALL:
        negative_outcome = Outcome.SOME
            
    for constraint_type in OUTCOME_TO_CONSTRAINTS[(positive_outcome, negative_outcome)]:
        if constraint_type == Con.GENERALISATION and not gens_pruned_flag:
            rules.update(constrainer.generalisation_constraint(program, before, min_clause))
            gens_pruned_flag = True
        elif constraint_type == Con.SPECIALISATION and not specs_pruned_flag:
            rules.update(constrainer.specialisation_constraint(program, before, min_clause))
            specs_pruned_flag = True
        elif constraint_type == Con.REDUNDANCY:
            rules.update(constrainer.redundancy_constraint(program, before, min_clause))
        elif constraint_type == Con.BANISH:
            rules.update(constrainer.banish_constraint(program, before, min_clause))

    if settings.noisy:
        if gens_pruned_flag:
            stats.can_prune_gens.append(stats.all_programs[-1])
        if specs_pruned_flag:
            stats.can_prune_specs.append(stats.all_programs[-1])

    if settings.functional_test and tester.is_non_functional(program):
        rules.update(constrainer.generalisation_constraint(program, before, min_clause))

    # eliminate generalisations of clauses that contain redundant literals
    for rule in tester.check_redundant_literal(program):
        rules.update(constrainer.redundant_literal_constraint(rule, before, min_clause))

    # eliminate generalisations of programs that contain redundant clauses
    if tester.check_redundant_clause(program):
        rules.update(constrainer.generalisation_constraint(program, before, min_clause))

    if len(program) > 1:
        # evaluate inconsistent sub-clauses
        for rule in program:
            if Clause.is_separable(rule) and tester.is_inconsistent(rule):
                for x in constrainer.generalisation_constraint([rule], before, min_clause):
                    rules.add(x)

        # eliminate totally incomplete rules
        if all(Clause.is_separable(rule) for rule in program):
            for rule in program:
                if tester.is_totally_incomplete(rule):
                    for x in constrainer.redundancy_constraint([rule], before, min_clause):
                        rules.add(x)

    stats.register_rules(rules)

    return rules

PROG_KEY = 'prog'

def calc_score(conf_matrix):
    tp, fn, tn, fp = conf_matrix
    return tp + tn

def popper(settings, stats):
    solver = ClingoSolver(settings)
    tester = Tester(settings)
    settings.num_pos, settings.num_neg = len(tester.pos), len(tester.neg)
    grounder = ClingoGrounder()
    constrainer = Constrain()
    best_score = None

    for size in range(1, settings.max_literals + 1):
        stats.update_num_literals(size)
        solver.update_number_of_literals(size)

        while not len(stats.all_programs) == settings.max_programs:
            # GENERATE HYPOTHESIS
            with stats.duration('generate'):
                model = solver.get_model()
                if not model:
                    break
                (program, before, min_clause) = generate_program(model)

            # TEST HYPOTHESIS
            with stats.duration('test'):
                conf_matrix = tester.test(program)
                outcome = decide_outcome(conf_matrix)
                score = calc_score(conf_matrix)

            stats.register_program(program, conf_matrix)

            # UPDATE BEST PROGRAM
            if best_score == None or score > best_score:
                best_score = score

                if outcome == (Outcome.ALL, Outcome.NONE):
                    stats.register_solution(program, conf_matrix)
                    return stats.solution.code

                stats.register_best_program(program, conf_matrix)

            # BUILD RULES
            with stats.duration('build'):
                rules = build_rules(settings, stats, constrainer, tester, program, before, min_clause, outcome)

            # GROUND RULES
            with stats.duration('ground'):
                rules = ground_rules(stats, grounder, solver.max_clauses, solver.max_vars, rules)

            # UPDATE SOLVER
            with stats.duration('add'):
                solver.add_ground_clauses(rules)

    stats.register_completion()
    return stats.best_program.code if stats.best_program else None

def show_hspace(settings):
    f = lambda i, m: print(f'% program {i}\n{format_program(generate_program(m)[0])}')
    ClingoSolver.get_hspace(settings, f)

def learn_solution(settings):
    stats = Stats(log_best_programs=settings.info)
    log_level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(level=log_level, stream=sys.stderr, format='%(message)s')
    timeout(popper, (settings, stats), timeout_duration=int(settings.timeout))

    if stats.solution:
        prog_stats = stats.solution
    elif stats.best_programs:
        prog_stats = stats.best_programs[-1]
    else:
        return None, stats

    return prog_stats.code, stats
