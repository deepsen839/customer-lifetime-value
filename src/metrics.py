import numpy as np
def retention_economics(eligible_customers,offer_cost_per_customer,incremental_lift,contribution_value_per_save):
    if eligible_customers<0 or offer_cost_per_customer<0 or contribution_value_per_save<0: raise ValueError("counts and costs must be non-negative")
    if not 0<=incremental_lift<=1: raise ValueError("lift must be between 0 and 1")
    cost=eligible_customers*offer_cost_per_customer
    saves=eligible_customers*incremental_lift
    value=saves*contribution_value_per_save
    net=value-cost
    # ROI is undefined when the denominator is zero; None keeps API JSON valid.
    roi=net/cost*100 if cost else None
    return {"offer_cost":cost,"incremental_saves":saves,"incremental_value":value,"net_benefit":net,"roi_pct":roi}
