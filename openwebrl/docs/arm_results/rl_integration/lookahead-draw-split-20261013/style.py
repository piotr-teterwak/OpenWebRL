"""Shared names, colours and panels for the draw-split figures."""
STYLE = {  # key in curve.json -> (legend label, colour, line style)
    "Best on observed draws": ("Oracle (observed draws)", "#1baf7a", "-"),
    "Lookahead K=3": ("Lookahead K=3", "#1f5fae", "-"),
    "Lookahead K=2": ("Lookahead K=2", "#4f8fdc", "-"),
    "Lookahead K=1": ("Lookahead K=1", "#93bdef", "-"),
    "Immediate result": ("Post-execution", "#eb6834", "-"),
    "Before execution": ("Pre-execution", "#7b4fd0", "-"),
    "WebArbiter formulation": ("WebArbiter (Luna)", "#8c6d31", "-."),
    "WebShepherd formulation": ("WebShepherd (Luna)", "#b5a33a", "-."),
    "PRO-CUA formulation": ("PRO-CUA (Luna)", "#4a8c8c", "-."),
    "Actor first sample": ("Actor (first sample)", "#b04a9b", ":"),
    "Uniform": ("Uniform", "#9a9890", "--"),
}
PANELS = [("pooled", "All"), ("lookahead-fast1k-20261013", "New states"), ("lookahead-v2-20261013", "Earlier states")]
TITLES = {"invalid_as_failure": "Invalid draws = failure", "valid_only": "Valid draws only"}
