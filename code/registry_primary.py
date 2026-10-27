"""Registered primary-endpoint results for the cystic fibrosis trials.

Reads the posted results of each trial from the ClinicalTrials.gov API v2 and records the
p-value of every posted analysis of the registered primary outcome. The point it supports
(Section 5.1, Q2): trials that met their primary endpoint are non-significant for the rare
serious-adverse-event outcome the network pools, while their weights depend only on arm sizes.

Usage: python3 registry_primary.py  ->  data/registry_primary_endpoints.csv
"""
import csv
import json
import os
import urllib.request

TRIALS = {
    "NCT00909532": "STRIVE", "NCT00909727": "ENVISION", "NCT01807923": "TRAFFIC",
    "NCT01807949": "TRANSPORT", "NCT02347657": "EVOLVE", "NCT02392234": "EXPAND",
    "NCT03525444": "VX17-445-102", "NCT03525548": "VX17-445-103",
}
API = "https://clinicaltrials.gov/api/v2/studies/{}?fields=resultsSection.outcomeMeasuresModule"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "registry_primary_endpoints.csv")


def primary_analyses(nct):
    with urllib.request.urlopen(API.format(nct), timeout=30) as r:
        d = json.load(r)
    rows = []
    for o in d["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"]:
        if o.get("type") != "PRIMARY":
            continue
        groups = {g["id"]: g["title"] for g in o.get("groups", [])}
        for a in o.get("analyses", []):
            rows.append({
                "nct": nct, "study": TRIALS[nct], "outcome": o["title"],
                "comparison": " vs ".join(groups.get(g, g) for g in a.get("groupIds", [])),
                "estimate": a.get("paramValue"), "method": a.get("statisticalMethod"),
                "p_value": a.get("pValue"),
            })
    return rows


if __name__ == "__main__":
    rows = [r for nct in TRIALS for r in primary_analyses(nct)]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['study']:13s} p={r['p_value']:>8s}  {r['comparison'][:70]}")
