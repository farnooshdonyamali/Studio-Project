from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessageGraph
from typing import TypedDict, List


# --- Node: Input Node ---
def input_node(state):
    return state


# --- Node: Processing Node ---
def process_metrics(state):
    data = state["input"]

    today = data["today"]
    yesterday = data["yesterday"]

    profit = today["revenue"] - today["cost"]
    prev_profit = yesterday["revenue"] - yesterday["cost"]

    percent_change_revenue = (
        (today["revenue"] - yesterday["revenue"]) / yesterday["revenue"]
    ) * 100
    percent_change_cost = (
        (today["cost"] - yesterday["cost"]) / yesterday["cost"]
    ) * 100

    today_cac = today["cost"] / today["customers"] if today["customers"] else 0
    yesterday_cac = (
        yesterday["cost"] / yesterday["customers"] if yesterday["customers"] else 0
    )

    cac_change_percent = (
        ((today_cac - yesterday_cac) / yesterday_cac) * 100 if yesterday_cac else 0
    )

    state["metrics"] = {
        "profit": profit,
        "prev_profit": prev_profit,
        "percent_change_revenue": percent_change_revenue,
        "percent_change_cost": percent_change_cost,
        "today_cac": today_cac,
        "yesterday_cac": yesterday_cac,
        "cac_change_percent": cac_change_percent,
    }
    return state


# --- Node: Recommendation Node ---
def recommendation_node(state):
    metrics = state["metrics"]
    recommendations = []
    alerts = []

    if metrics["profit"] < 0:
        alerts.append("Warning: Negative profit.")
        recommendations.append("Reduce costs if profit is negative.")

    if metrics["cac_change_percent"] > 20:
        alerts.append("Warning: CAC increased more than 20%.")
        recommendations.append("Review marketing campaigns.")

    if metrics["percent_change_revenue"] > 10:
        recommendations.append(
            "Consider increasing advertising budget if sales are growing."
        )

    state["output"] = {
        "profit": metrics["profit"],
        "alerts": alerts,
        "recommendations": recommendations,
    }
    return state


# --- LangGraph Setup ---
class AgentState(TypedDict):
    input: dict
    metrics: dict
    output: dict


graph = StateGraph(AgentState)
graph.add_node("input", input_node)
graph.add_node("process", process_metrics)
graph.add_node("recommend", recommendation_node)

graph.set_entry_point("input")
graph.add_edge("input", "process")
graph.add_edge("process", "recommend")
graph.add_edge("recommend", END)

ai_agent = graph.compile()

# --- Sample Run ---
if __name__ == "__main__":
    sample_input = {
        "input": {
            "today": {"revenue": 12000, "cost": 9000, "customers": 300},
            "yesterday": {"revenue": 10000, "cost": 7000, "customers": 350},
        }
    }

    result = ai_agent.invoke(sample_input)
    print(result["output"])


# --- Test ---
def test_agent():
    test_input = {
        "input": {
            "today": {"revenue": 12000, "cost": 9000, "customers": 300},
            "yesterday": {"revenue": 10000, "cost": 7000, "customers": 350},
        }
    }
    expected_alerts = ["Warning: CAC increased more than 20%."]
    expected_recs = [
        "Review marketing campaigns.",
        "Consider increasing advertising budget if sales are growing.",
    ]

    out = ai_agent.invoke(test_input)["output"]

    assert out["profit"] == 3000
    assert set(out["alerts"]).intersection(set(expected_alerts))
    assert set(out["recommendations"]).intersection(set(expected_recs))
    print("Test passed!")


# Run test
if __name__ == "__main__":
    test_agent()
