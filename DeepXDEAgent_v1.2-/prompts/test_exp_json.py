import json
import os

with open("exp.json", "r") as f:
    exp_json = json.load(f)
    
test_exp_json = f"""
{{
    "Standard_format": "I would like to solve the following PDE equation: specific_PDE_equation\nwhere specific_parameters\nThe computational geometry domain / time doamin is specific_geometry_domain_and_time_domain\nThe initial condition specific_initial_condition\nThe boundary condition specific_boundary_condition\nothers",
    "Example": "I would like to solve the following Allen-Cahn equation: \\frac{{\\partial u}}{{\\partial t}} = d\\frac{{\\partial^2 u}}{{\\partial x^2}} + 5(u - u^3)\nwhere d=0.01\nThe computational geometry domain is [-1,1], and the time domain is [0,1]\nThe initial condition u(x,0) = x^2\\cos(\\pi x)\nThe Dirichlet boundary condition u(-1,t) = u(1,t) = -1"
}}
"""

test_exp_json_dict = json.loads(test_exp_json)

print(exp_json)
print(test_exp_json_dict)

assert exp_json == test_exp_json_dict, "exp.json is not equal to the test_exp_json_dict"

