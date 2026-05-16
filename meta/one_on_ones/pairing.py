def validate_make_pairing_graph_inputs(
    names: list[str],
    rounds: int,
    preferences: dict[tuple[str, str], int],
    ta_group: set[str],
    unavailability: dict[str, set[int]] | None = None,
) -> list[str]:
    errors = []
    for pair in preferences.keys():
        if pair[0] not in names:
            errors.append(f"Pair {pair} contains a name not in the names list: {pair[0]}")
        if pair[1] not in names:
            errors.append(f"Pair {pair} contains a name not in the names list: {pair[1]}")

    for name in set(names):
        count = names.count(name)
        if count > 1:
            errors.append(f"Name {name} appears {count} times in the names list")

    for pair in preferences.keys():
        if len(pair) != 2:
            errors.append(f"Preference pair {pair} is not a pair")

    for name in ta_group:
        if name not in names:
            errors.append(f"TA name {name} is not in the names list")

    for a, b in preferences:
        if (b, a) in preferences and preferences[b, a] != preferences[a, b]:
            errors.append(
                f"Preference {a, b} is not symmetric: {preferences[a, b]} != {preferences[b, a]}"
            )

    for name in unavailability or {}:
        if name not in names:
            errors.append(f"Unavailability entry for unknown participant: {name}")

    return errors


def make_pairing_graph(
    # names: list[str], rounds: int, forced_pairs: list[tuple[str, str]]
    names: list[str],
    rounds: int,
    preferences: dict[tuple[str, str], int],
    ta_group: set[str],
    ta_group_weight: int = 2,
    unavailability: dict[str, set[int]] | None = None,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """
    Generate a pairing schedule for a given list of names over a specified number of rounds,
    taking into account preferences and a special group of names (TA group).

    Args:
        names (list[str]): A list of unique names to be paired.
        rounds (int): The number of rounds for which the pairings should be generated.
        preferences (dict[tuple[str, str], int]): A dictionary where keys are tuples of names
            representing pairs, and values are integers representing the weight of the preference
            for that pair. A weight of 0 means the pair should not be matched.
        ta_group (set[str]): A set of names that belong to a special group (TA group).
            These names have special pairing rules.
        ta_group_weight (int, optional): The weight to be assigned to edges between names in the
            TA group and other names. Defaults to 2.
        unavailability (dict[str, set[int]] | None): Maps a participant name to the set of
            round indices (0-based) in which they are unavailable. Unavailable participants
            receive "-" for that round and their remaining edges are preserved for future rounds.

    Returns:
        dict[str, list[str]]: A dictionary where keys are names and values are lists of names
            representing the pairings for each round. Each list has a length equal to the number
            of rounds, and the ith element in the list is the name of the person paired with the
            key name in the ith round.

    Raises:
        AssertionError: If any of the following conditions are not met:
            - All names in the preference pairs are in the names list.
            - All names in the names list are unique.
            - All preference pairs are of length 2.
            - All names in the TA group are in the names list.

    Special Rules:
    - TA Group: Names in the `ta_group` have special pairing rules.
        People outside of the TA group will meet with a member of the group only when everyone
        else has met with the group as many times as them.
        Members of the TA group cannot meet between themselves.
    - If the number of names is odd, a dummy name "-" is added to the list of names.
        And we rotate who is not paired with anyone.
    """

    errors = validate_make_pairing_graph_inputs(
        names, rounds, preferences, ta_group, unavailability
    )
    if errors:
        raise ValueError("\n".join(errors))

    import networkx as nx
    import random as _random

    rng = _random.Random(seed)

    meets_from_group = {name: 0 for name in names}
    # Always add a dummy "-" node so that any round whose available subgraph has
    # odd cardinality (e.g. due to unavailability) can include it and rotate
    # who gets left out. The matched "-" edge is removed each round, which
    # forces "-" to pair with a different person next time.
    names_with_dummy = names + ["-"]

    G = nx.Graph()
    G.add_nodes_from(names_with_dummy)
    # We make a complete graph
    G.add_edges_from(
        (name, match) for name in names_with_dummy for match in names_with_dummy if name != match
    )
    # We remove or update edges based on preferences
    for (a, b), weight in preferences.items():
        if weight == 0:
            G.remove_edge(a, b)
        else:
            G[a][b]["weight"] = weight
    # Remove edges between members of the TA group
    G.remove_edges_from((a, b) for a in ta_group for b in ta_group if a != b)

    all_pairings: dict[str, list[str]] = {name: ["-"] * rounds for name in names_with_dummy}

    for round in range(rounds):
        # Adapt the weight of edges between non-meet group members to meetgroup
        # They should meet with the group iff everyone else as met as much as them with the group
        max_meet_number = max(meets_from_group.values())
        for name in names:
            if name in ta_group:
                continue

            for meet in ta_group:
                if name != "-" and G.has_edge(name, meet):
                    default_edge_weight = preferences.get(
                        (name, meet), preferences.get((meet, name))
                    )
                    if default_edge_weight is None:
                        G[name][meet]["weight"] = (
                            ta_group_weight if meets_from_group[name] < max_meet_number else 1
                        )

        # Exclude unavailable participants via a subgraph view (their edges survive in G).
        # Include the dummy "-" only when the available set is odd, so it absorbs
        # exactly one person per such round and rotates via edge removal.
        unavailable_this_round = {
            name for name, rounds_set in (unavailability or {}).items() if round in rounds_set
        }
        available = [n for n in names if n not in unavailable_this_round]
        if len(available) % 2 == 1:
            available.append("-")
        graph_for_round = G.subgraph(available)

        # Jitter weights on a copy to break ties randomly without polluting G or
        # overriding the real preference / TA-balancing weights (jitter << 1).
        jittered = graph_for_round.copy()
        for u, v in jittered.edges():
            jittered[u][v]["weight"] = jittered[u][v].get("weight", 1) + rng.uniform(0, 1e-3)

        # Get a maximum matching
        matching: set[tuple[str, str]] = nx.max_weight_matching(jittered)
        # Remove the edges in the matching
        G.remove_edges_from(matching)
        # Add the matching to the pairings
        for a, b in matching:
            all_pairings[a][round] = b
            all_pairings[b][round] = a

            # Update the number of meetings with the group (dummy "-" doesn't count)
            if a in ta_group and b != "-":
                meets_from_group[b] += 1
            elif b in ta_group and a != "-":
                meets_from_group[a] += 1

    all_pairings.pop("-", None)
    return all_pairings
